#!/usr/bin/env python3
"""Export Sparse4D+HiVT handoff packets (JSON + NPZ) for downstream control/risk.

Draft format for teammate feedback — not a frozen schema.
"""
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
SCHEMA_VERSION = 'handoff_packet_v0'


def arrays_to_temporal(arrays: Dict[str, Any]) -> TemporalData:
    skip = {
        'track_ids', 'prediction_eligible', 'episode_id', 'end_timestamp',
        'y',
        'map_source', 'map_n_lane_vectors', 'map_n_lane_actor_edges',
    }
    kwargs: Dict[str, Any] = {}
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
        elif isinstance(v, (float, int, str)):
            kwargs[k] = v
    kwargs['num_nodes'] = int(arrays['num_nodes'])
    return TemporalData(**kwargs)


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _lane_export(arrays: Dict[str, Any], max_vectors: int = 400) -> Dict[str, Any]:
    lv = np.asarray(arrays.get('lane_vectors', np.zeros((0, 2))), dtype=np.float32)
    # lane_vectors in AV2 adapter are often stored as local vectors; also export counts.
    n = int(lv.shape[0])
    take = min(n, max_vectors)
    out: Dict[str, Any] = {
        'map_source': arrays.get('map_source'),
        'n_lane_vectors': int(arrays.get('map_n_lane_vectors', n)),
        'n_lane_actor_edges': int(arrays.get('map_n_lane_actor_edges', 0)),
        'note': (
            'Lane tensors follow AV2 HiVT local-encoder contract (scene-local). '
            'Full town package: integration/maps/TownXX_lanes_rh.npz'
        ),
    }
    if take > 0:
        # Prefer absolute-ish positions if 'positions' style fields exist; else vectors head.
        pos = arrays.get('lane_positions')
        if pos is None:
            # Many windows only have lane_vectors; still useful as relative segments.
            out['lane_vectors_head'] = lv[:take].tolist()
            out['lane_vectors_head_truncated'] = take < n
        else:
            pos_a = np.asarray(pos, dtype=np.float32)
            out['lane_positions_head'] = pos_a[:take].tolist()
            out['lane_positions_head_truncated'] = take < int(pos_a.shape[0])
    return out


def build_window_packet(
    wi: int,
    arrays: Dict[str, Any],
    world: np.ndarray,
    pi_np: np.ndarray,
    y_np: np.ndarray,
    hivt_lat_s: float,
    sparse4d_latency_s: float,
    poses_rows: List[Dict[str, Any]],
    synced: Dict[str, Any],
    risk: Dict[str, Any],
) -> Dict[str, Any]:
    t_obs = float(arrays['end_timestamp'])
    track_ids = [str(t) for t in arrays['track_ids']]
    actor_type = np.asarray(arrays['actor_type']).astype(int)
    type_to_name = {0: 'car', 1: 'pedestrian', 2: 'motorcycle', 3: 'bicycle', 4: 'bus', 5: 'unknown'}
    av_index = int(arrays['av_index'])
    agent_index = int(arrays['agent_index'])

    # HiVT raw futures: [F,N,H,2] modes x agents x horizon x xy
    f, n, h, _ = world.shape
    abs_times = (t_obs + (np.arange(h) + 1) * 0.1).tolist()
    agents = []
    for i, tid in enumerate(track_ids):
        agents.append({
            'track_id': tid,
            'actor_type_id': int(actor_type[i]),
            'actor_type': type_to_name.get(int(actor_type[i]), 'unknown'),
            'is_av': bool(i == av_index),
            'is_focal_agent': bool(i == agent_index),
            'mode_probabilities': pi_np[i].tolist(),
            'future_world_xy_modes': world[:, i].tolist(),  # [F,H,2]
            'future_abs_times_from_t_obs': abs_times,
        })

    aligned = risk.get('aligned_at_t_fuse') or {}
    hivt_al = aligned.get('hivt') or {}
    s4d_al = aligned.get('sparse4d') or {}
    b2d_al = aligned.get('bench2drive') or {}

    synced_xy = synced['synced_world_xy']
    # synced may drop leading steps → [F,N,H',2]
    agents_synced = []
    fut_times = list(hivt_al.get('future_abs_times') or [])
    for i, tid in enumerate(track_ids):
        agents_synced.append({
            'track_id': tid,
            'mode_probabilities': pi_np[i].tolist(),
            'future_world_xy_modes': synced_xy[:, i].tolist() if synced_xy.size else [],
            'future_abs_times': fut_times,
        })

    return {
        'schema_version': SCHEMA_VERSION,
        'window_index': wi,
        'episode_id': arrays['episode_id'],
        'coordinate_frame': 'world_rh',
        'units': {'distance': 'm', 'time': 's', 'velocity': 'm/s', 'angle': 'rad'},
        'dt_s': 0.1,
        'detector_hz': 10,
        't_obs': t_obs,
        't_fuse': float(risk['t_fuse']),
        'latencies': risk.get('latencies'),
        'fusion_clock': risk.get('fusion_clock'),
        'same_clock': bool(aligned.get('same_clock')),
        'ego': {
            'av_index': av_index,
            'track_id': track_ids[av_index] if 0 <= av_index < len(track_ids) else None,
            'at_obs': (b2d_al.get('ego_at_obs') or {}),
            'at_fuse': (b2d_al.get('ego_at_fuse') or {}),
        },
        'objects_at_fuse': s4d_al.get('objects_at_fuse') or [],
        'hivt_raw_at_t_obs': {
            'shape_FNH2': [int(f), int(n), int(h), 2],
            'n_modes': int(f),
            'n_agents': int(n),
            'horizon_steps': int(h),
            'agents': agents,
            'note': 'Futures anchored at T_obs; step k ≈ T_obs + (k+1)*0.1s',
        },
        'hivt_aligned_at_t_fuse': {
            'skipped_future_steps': hivt_al.get('skipped_future_steps'),
            'remaining_future_steps': hivt_al.get('remaining_future_steps'),
            'correct_by': hivt_al.get('correct_by'),
            'agents': agents_synced,
            'note': 'Use this for risk/control — past steps relative to T_fuse already removed',
        },
        'map': _lane_export(arrays),
        'risk_summary': summarize_aligned_packet(risk),
        'model_meta': {
            'y_hat_shape': list(y_np.shape),
            'pi_shape': list(pi_np.shape),
            'hivt_latency_s': hivt_lat_s,
            'sparse4d_latency_s_assumed_or_measured': sparse4d_latency_s,
        },
        'feedback_wanted': [
            'Prefer JSON-only, NPZ-only, or both?',
            'Need full horizon or only first 2.0s (20 steps)?',
            'Need all modes or top-1 / top-k only?',
            'Map: local lane vectors enough, or world polylines / markings?',
            'Object boxes: center+vel enough, or full lwh/yaw required?',
        ],
    }


def write_readme(out_dir: Path) -> None:
    text = f"""# Handoff packet (draft) — Sparse4D + HiVT

Schema: `{SCHEMA_VERSION}` (피드백용 초안, 고정 포맷 아님)

## 목적

위험 판단 / 후보 경로 / 제어 쪽에서 바로 열어볼 수 있도록
**자차·주변 객체·HiVT 미래 궤적·(요약) 맵·시간 동기화**를 JSON + NPZ로 같이 제공합니다.

## 파일

| 파일 | 설명 |
|------|------|
| `manifest.json` | 에피소드·윈도우 목록·생성 메타 |
| `windows/window_XX.json` | 사람이 읽기 쉬운 통합 패킷 |
| `windows/window_XX.npz` | 배열 중심 (numpy 로드) |
| `SCHEMA.md` | 필드 의미 |
| `README.md` | 이 파일 |

## 시계

- `t_obs`: Sparse4D/HiVT가 본 프레임 시각
- `t_fuse`: 위험 판단 소비 시각 (= 기본 `t_obs + L_s4d + L_hivt`)
- **제어/위험은 `hivt_aligned_at_t_fuse` + `objects_at_fuse` + `ego.at_fuse`를 사용**

자세한 규칙: `integration/docs/TIME_SYNC.md`

## NPZ 키

- `world_xy`: float32 `[F,N,H,2]` HiVT world XY (T_obs 기준)
- `synced_world_xy`: float32 `[F,N,H',2]` T_fuse 이후만
- `pi`: float32 `[N,F]` mode 확률
- `future_abs_times_raw`: float64 `[H]` T_obs 기준 absolute times
- `future_abs_times_synced`: float64 `[H']`
- `track_ids`: unicode `[N]`
- `actor_type`: int64 `[N]`
- `objects_center_xy_at_fuse`: float64 `[M,2]`
- `ego_xy_at_fuse`: float64 `[2]`
- `t_obs`, `t_fuse`, `sparse4d_latency_s`, `hivt_latency_s`: scalars

## 피드백 부탁

JSON/NPZ 중 선호, horizon/mode 축소 여부, 맵·박스 필드를 알려주세요.
"""
    (out_dir / 'README.md').write_text(text, encoding='utf-8')
    schema = """# SCHEMA — handoff_packet_v0

## Top-level JSON fields

| field | meaning |
|-------|---------|
| `schema_version` | `handoff_packet_v0` |
| `episode_id` | Bench2Drive episode id |
| `window_index` | export order |
| `coordinate_frame` | `world_rh` (Bench2Drive RH world, meters) |
| `t_obs` / `t_fuse` | seconds |
| `latencies.sparse4d_s` / `hivt_s` / `total_s` | seconds |
| `ego.at_fuse.ego_xy` | `[x,y]` ego at T_fuse |
| `objects_at_fuse[]` | track_id, class, center_xy_at_fuse, velocity_xy |
| `hivt_raw_at_t_obs.agents[]` | per-agent modes × future XY from T_obs |
| `hivt_aligned_at_t_fuse.agents[]` | same, steps before T_fuse removed |
| `map` | lane vector summary / counts (local HiVT map crop) |
| `risk_summary` | compact alignment status |

## Actor type ids

0 car, 1 pedestrian, 2 motorcycle, 3 bicycle, 4 bus, 5 unknown

## HiVT shapes

- modes `F` (usually 6)
- agents `N`
- horizon `H` (AV2 HiVT fut=60 → 6.0s at 0.1s)
- XY last dim = 2

For emergency planning you typically only need the first ~20 steps (2.0s).
"""
    (out_dir / 'SCHEMA.md').write_text(schema, encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str,
                   default=str(DEFAULT_HIVT_CKPT))
    p.add_argument('--episode', type=str, default=DEFAULT_EP)
    p.add_argument('--predictions', type=str,
                   default=str(FIXTURES / 'predictions' / DEFAULT_EP / 'predictions.jsonl'))
    p.add_argument('--poses', type=str,
                   default=str(FIXTURES / 'poses' / DEFAULT_EP / 'poses_10hz.json'))
    p.add_argument('--max_windows', type=int, default=5)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--sparse4d_latency_s', type=float, default=0.140)
    p.add_argument('--output_dir', type=str, default='')
    args = p.parse_args()

    pred_path = Path(args.predictions)
    poses_path = Path(args.poses)
    ckpt = Path(args.ckpt_path)
    for path in (pred_path, poses_path, ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir = Path(args.output_dir) if args.output_dir else (
        OUTPUTS_ROOT / f'handoff_packet_{stamp}'
    )
    win_dir = out_dir / 'windows'
    win_dir.mkdir(parents=True, exist_ok=True)
    write_readme(out_dir)

    device = torch.device(
        args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu'
    )
    diagnostics: Dict[str, Any] = {}
    poses_rows = json.loads(poses_path.read_text(encoding='utf-8'))
    windows = list(iter_av2_windows(
        args.episode, pred_path, poses_path,
        max_windows=args.max_windows, diagnostics=diagnostics,
    ))

    if not windows:
        raise RuntimeError(f'No windows: {diagnostics}')

    model = HiVT.load_from_checkpoint(checkpoint_path=str(ckpt))
    model.enable_fast_inference()
    model.eval().to(device)

    # warmup
    warm = arrays_to_temporal(windows[0]).to(device)
    warm.y = None
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _ = model(warm)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    written: List[Dict[str, Any]] = []
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

        packet = build_window_packet(
            wi=wi,
            arrays=arrays,
            world=world,
            pi_np=pi_np,
            y_np=y_np,
            hivt_lat_s=hivt_lat_s,
            sparse4d_latency_s=args.sparse4d_latency_s,
            poses_rows=poses_rows,
            synced=synced,
            risk=risk,
        )
        json_path = win_dir / f'window_{wi:02d}.json'
        json_path.write_text(
            json.dumps(_jsonable(packet), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

        aligned = risk.get('aligned_at_t_fuse') or {}
        objs = (aligned.get('sparse4d') or {}).get('objects_at_fuse') or []
        centers = np.zeros((0, 2), dtype=np.float64)
        if objs:
            centers = np.asarray([o['center_xy_at_fuse'] for o in objs], dtype=np.float64)
        ego_xy = np.asarray(
            ((aligned.get('bench2drive') or {}).get('ego_at_fuse') or {}).get('ego_xy') or [np.nan, np.nan],
            dtype=np.float64,
        )
        h = world.shape[2]
        abs_times_raw = t_obs + (np.arange(h) + 1) * 0.1
        fut_synced = np.asarray(
            (aligned.get('hivt') or {}).get('future_abs_times') or [], dtype=np.float64
        )
        npz_path = win_dir / f'window_{wi:02d}.npz'
        np.savez_compressed(
            npz_path,
            world_xy=world.astype(np.float32),
            synced_world_xy=synced['synced_world_xy'].astype(np.float32),
            pi=pi_np.astype(np.float32),
            y_hat=y_np.astype(np.float32),
            future_abs_times_raw=abs_times_raw.astype(np.float64),
            future_abs_times_synced=fut_synced,
            track_ids=np.asarray(arrays['track_ids']),
            actor_type=np.asarray(arrays['actor_type'], dtype=np.int64),
            objects_center_xy_at_fuse=centers,
            ego_xy_at_fuse=ego_xy,
            t_obs=np.asarray([t_obs], dtype=np.float64),
            t_fuse=np.asarray([float(risk['t_fuse'])], dtype=np.float64),
            sparse4d_latency_s=np.asarray([args.sparse4d_latency_s], dtype=np.float64),
            hivt_latency_s=np.asarray([hivt_lat_s], dtype=np.float64),
            av_index=np.asarray([int(arrays['av_index'])], dtype=np.int64),
            agent_index=np.asarray([int(arrays['agent_index'])], dtype=np.int64),
        )

        written.append({
            'window_index': wi,
            't_obs': t_obs,
            't_fuse': float(risk['t_fuse']),
            'json': str(json_path.relative_to(out_dir)),
            'npz': str(npz_path.relative_to(out_dir)),
            'n_agents': int(arrays['num_nodes']),
            'hivt_latency_ms': hivt_ms,
            'same_clock': bool(aligned.get('same_clock')),
        })
        print(
            f'window[{wi}] T_obs={t_obs:.1f} T_fuse={float(risk["t_fuse"]):.3f} '
            f'N={arrays["num_nodes"]} HiVT={hivt_ms:.1f}ms -> {json_path.name} + {npz_path.name}'
        )

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'created_utc': stamp,
        'episode_id': args.episode,
        'ckpt': str(ckpt),
        'predictions': str(pred_path),
        'poses': str(poses_path),
        'sparse4d_latency_s': args.sparse4d_latency_s,
        'device': str(device),
        'coverage': diagnostics,
        'windows': written,
        'notes': [
            'Draft handoff for control/risk teammate feedback.',
            'Prefer hivt_aligned_at_t_fuse + objects_at_fuse + ego.at_fuse.',
            'Fixtures detections (or live Sparse4D jsonl) → HiVT → risk_fusion.',
        ],
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )
    print(f'wrote {out_dir}')
    print(json.dumps({'ok': True, 'n_windows': len(written), 'out': str(out_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
