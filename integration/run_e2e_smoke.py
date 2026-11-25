#!/usr/bin/env python3
"""End-to-end smoke: saved Sparse4D dets → AV2 HiVT → time-synced world forecasts."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from integration.paths import (
    DEFAULT_HIVT_CKPT,
    INTEGRATION_ROOT,
    OUTPUTS_ROOT,
    REPO_ROOT as ROOT,
    ensure_import_paths,
)

ensure_import_paths()

from integration.av2_adapter import world_predictions_av2
from integration.bridge import iter_av2_windows
from integration.risk_fusion import (
    build_risk_packet,
    detections_from_hivt_window,
    summarize_aligned_packet,
)
from integration.time_sync import sync_forecast
from models.hivt import HiVT
from utils import TemporalData

DEFAULT_EP = 'DynamicObjectCrossing_Town12_Route17906_Weather0_02-06-18-12-57'
FIXTURES = INTEGRATION_ROOT / 'fixtures'


def arrays_to_temporal(arrays: Dict[str, Any]) -> TemporalData:
    skip = {
        'track_ids', 'prediction_eligible', 'episode_id', 'end_timestamp',
        'y',  # None
        'map_source', 'map_n_lane_vectors', 'map_n_lane_actor_edges',
    }
    kwargs = {}
    for k, v in arrays.items():
        if k in skip or v is None:
            continue
        if isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                kwargs[k] = torch.from_numpy(v)
            elif np.issubdtype(v.dtype, np.floating):
                kwargs[k] = torch.from_numpy(v.astype(np.float32))
            elif np.issubdtype(v.dtype, np.integer):
                kwargs[k] = torch.from_numpy(v.astype(np.int64))
            else:
                kwargs[k] = torch.from_numpy(v)
        elif isinstance(v, (float, int)):
            kwargs[k] = v
        elif isinstance(v, str):
            kwargs[k] = v
    kwargs['num_nodes'] = int(arrays['num_nodes'])
    return TemporalData(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_path', type=str,
        default=str(DEFAULT_HIVT_CKPT),
    )
    parser.add_argument('--episode', type=str, default=DEFAULT_EP)
    parser.add_argument(
        '--predictions', type=str,
        default=str(FIXTURES / 'predictions' / DEFAULT_EP / 'predictions.jsonl'),
    )
    parser.add_argument(
        '--poses', type=str,
        default=str(FIXTURES / 'poses' / DEFAULT_EP / 'poses_10hz.json'),
    )
    parser.add_argument('--max_windows', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--sparse4d_latency_s', type=float, default=0.146,
        help='Sparse4D latency (default: measured mean on RTX 4080 Laptop, live smoke)',
    )
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    poses_path = Path(args.poses)
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    if not poses_path.is_file():
        raise FileNotFoundError(poses_path)
    ckpt = Path(args.ckpt_path)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu')
    out_dir = Path(args.output_dir) if args.output_dir else (
        OUTPUTS_ROOT / f'e2e_smoke_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print('========== E2E SMOKE ==========')
    print(f'episode        : {args.episode}')
    print(f'predictions    : {pred_path}')
    print(f'poses          : {poses_path}')
    print(f'ckpt           : {ckpt}')
    print(f'device         : {device}')
    print(f'sparse4d_lat_s : {args.sparse4d_latency_s} (assumed)')
    print(f'output         : {out_dir}')

    diagnostics: Dict[str, Any] = {}
    poses_rows = json.loads(Path(poses_path).read_text(encoding='utf-8'))
    windows = list(iter_av2_windows(
        args.episode, pred_path, poses_path,
        max_windows=args.max_windows, diagnostics=diagnostics,
    ))
    (out_dir / 'coverage.json').write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    print(f'windows        : {len(windows)}  rejections={diagnostics.get("rejection_counts")}')
    if not windows:
        raise RuntimeError('No eligible HiVT windows; see coverage.json')

    model = HiVT.load_from_checkpoint(checkpoint_path=str(ckpt))
    model.enable_fast_inference()
    model.eval().to(device)
    print(f'hivt hist/fut  : {model.historical_steps}/{model.future_steps}  '
          f'actor_type={getattr(model.hparams, "use_actor_type", None)}  '
          f'map_mode={getattr(model.hparams, "map_mode", None)}')

    # Warmup
    warm = arrays_to_temporal(windows[0]).to(device)
    warm.y = None
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _ = model(warm)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    results: List[Dict[str, Any]] = []
    for wi, arrays in enumerate(windows):
        data = arrays_to_temporal(arrays).to(device)
        data.y = None
        if device.type == 'cuda':
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            with torch.no_grad():
                y_hat, pi = model(data)
            ender.record()
            torch.cuda.synchronize()
            hivt_ms = float(starter.elapsed_time(ender))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                y_hat, pi = model(data)
            hivt_ms = (time.perf_counter() - t0) * 1000.0

        y_np = y_hat.detach().cpu().numpy()
        pi_np = torch.softmax(pi, dim=-1).detach().cpu().numpy()
        world = world_predictions_av2(y_np, arrays)
        t_obs = float(arrays['end_timestamp'])
        hivt_lat_s = hivt_ms / 1000.0
        synced = sync_forecast(
            observation_time=t_obs,
            world_xy=world,
            mode_probabilities=pi_np,
            sparse4d_latency_s=args.sparse4d_latency_s,
            hivt_latency_s=hivt_lat_s,
        )
        risk = build_risk_packet(
            t_obs=t_obs,
            sparse4d_latency_s=args.sparse4d_latency_s,
            hivt_latency_s=hivt_lat_s,
            b2d_payload={
                'episode_id': arrays['episode_id'],
                'av_index': int(arrays['av_index']),
            },
            sparse4d_payload={
                'n_tracks': int(arrays['num_nodes']),
                'track_ids': list(arrays['track_ids']),
                'objects': detections_from_hivt_window(arrays),
            },
            hivt_world_xy=world,
            hivt_mode_probabilities=pi_np,
            poses=poses_rows,
        )

        agent = int(arrays['agent_index'])
        case = {
            'window_index': wi,
            'episode_id': arrays['episode_id'],
            'observation_time': arrays['end_timestamp'],
            'num_nodes': int(arrays['num_nodes']),
            'av_index': int(arrays['av_index']),
            'agent_index': agent,
            'agent_track_id': arrays['track_ids'][agent],
            'y_hat_shape': list(y_np.shape),
            'pi_shape': list(pi_np.shape),
            'hivt_latency_ms': hivt_ms,
            'time_sync': synced['time_sync'],
            'risk_packet': summarize_aligned_packet(risk),
            'agent_mode_probabilities': pi_np[agent].tolist(),
            'agent_synced_world_xy_mode0_head': (
                synced['synced_world_xy'][0, agent, :5].tolist()
                if synced['synced_world_xy'].shape[2] >= 1 else []
            ),
            'finite_ok': bool(np.isfinite(world).all() and np.isfinite(pi_np).all()),
        }
        results.append(case)
        print(
            f'window[{wi}] T_obs={arrays["end_timestamp"]:.1f}s  N={arrays["num_nodes"]}  '
            f'HiVT={hivt_ms:.2f}ms  skip={synced["time_sync"]["skipped_future_steps"]}  '
            f'remain={synced["time_sync"]["remaining_future_steps"]}  finite={case["finite_ok"]}'
        )

        np.savez_compressed(
            out_dir / f'window_{wi:02d}.npz',
            y_hat=y_np,
            pi=pi_np,
            world_xy=world,
            synced_world_xy=synced['synced_world_xy'],
            actor_type=arrays['actor_type'],
            track_ids=np.asarray(arrays['track_ids']),
            end_timestamp=np.asarray([arrays['end_timestamp']]),
        )

    summary = {
        'ok': all(r['finite_ok'] for r in results),
        'num_windows': len(results),
        'ckpt': str(ckpt),
        'episode': args.episode,
        'assumed_sparse4d_latency_s': args.sparse4d_latency_s,
        'mean_hivt_latency_ms': float(np.mean([r['hivt_latency_ms'] for r in results])),
        'coverage': diagnostics,
        'cases': results,
        'notes': [
            'Sparse4D stage uses integration/fixtures JSONL (or live run_live_sparse4d.py).',
            'HiVT is AV2-trained checkpoint from this repo (hist=50, fut=60).',
            'risk_packet.aligned_at_t_fuse puts B2D ego, Sparse4D boxes, HiVT on T_fuse.',
        ],
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    print('-----------------------------------')
    print(f'PASS={summary["ok"]}  windows={summary["num_windows"]}  '
          f'mean_hivt_ms={summary["mean_hivt_latency_ms"]:.2f}')
    print(f'wrote {out_dir / "summary.json"}')
    if not summary['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
