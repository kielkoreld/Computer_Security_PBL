"""Bench2Drive prepared clip → Sparse4D → AV2 HiVT E2E (HiVT conda only).

Measures end-to-end latency per detection frame and Sparse4D detection AP
against gt_common.jsonl. HiVT forecast ADE needs future GT alignment (optional).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from integration.paths import (
    DEFAULT_HIVT_CKPT,
    DEFAULT_SPARSE4D_CKPT,
    OUTPUTS_ROOT,
    REPO_ROOT as REPO,
    SPARSE4D_ROOT,
    WEIGHTS_NOTES as WEIGHT_HANDOFF,
    ensure_import_paths,
    ensure_sparse4d_paths,
)

SPARSE4D_CKPT = DEFAULT_SPARSE4D_CKPT
DEFAULT_EP = 'DynamicObjectCrossing_Town12_Route17906_Weather0_02-06-18-12-57'
DEFAULT_PREP = SPARSE4D_ROOT / 'external_data' / 'prepared_10hz' / DEFAULT_EP
DEFAULT_DT = 0.1


def _disable_daf(obj) -> None:
    if isinstance(obj, dict):
        if 'use_deformable_func' in obj:
            obj['use_deformable_func'] = False
        for v in obj.values():
            _disable_daf(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _disable_daf(v)


def run_sparse4d(prep: Path, episode: str, device: str, warmup: int = 2) -> Dict[str, Any]:
    ensure_sparse4d_paths()
    from mmcv import Config
    from mmdet.models import build_detector
    import projects.mmdet3d_plugin  # noqa: F401
    from model_inputs import export_boxes

    prepared_q = prep / 'prepared'
    converted = prep / 'converted' / episode
    ckpt = SPARSE4D_CKPT
    if not ckpt.is_file():
        raise FileNotFoundError(
            f'Missing 10Hz Sparse4D weight: {ckpt}. '
            f'Copy from {WEIGHT_HANDOFF / "Sparse4Dv3_R50_Bench2Drive_10Hz_finetuned_best_additional_epoch_014.pth"}'
        )
    anchor = SPARSE4D_ROOT / 'handoff' / 'artifacts' / 'nuscenes_kmeans900_from_checkpoint.npy'
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    anchor.parent.mkdir(parents=True, exist_ok=True)
    np.save(anchor, sd['head.instance_bank.anchor'].detach().cpu().numpy())

    cfg = Config.fromfile(
        str(SPARSE4D_ROOT / 'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py')
    )
    cfg.model.img_backbone.pretrained = None
    cfg.model.head.instance_bank.anchor = str(anchor)
    # 10Hz finetune contract (engine.build default_dt=0.1).
    cfg.model.head.instance_bank.default_time_interval = DEFAULT_DT
    _disable_daf(cfg.model)

    model = build_detector(cfg.model).eval().to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    lut = torch.from_numpy(np.load(prepared_q / 'cache' / 'normalization_lut.npy')).to(device)
    channels = torch.arange(3, device=device)[None, None, :, None, None]
    with (prepared_q / 'cache' / episode / 'meta.pkl').open('rb') as f:
        meta = pickle.load(f)
    with (converted / 'sparse4d_infos.pkl').open('rb') as f:
        infos = pickle.load(f)['infos']
    rgb = np.load(prepared_q / 'cache' / episode / 'rgb.npy', mmap_mode='r')
    assert len(meta) == len(infos) == len(rgb)

    def one(i: int):
        raw = torch.from_numpy(np.array(rgb[i : i + 1], copy=True)).to(device)
        m = meta[i]
        return dict(
            img=lut[channels, raw.long()],
            projection_mat=torch.from_numpy(m['projection_mat'][None]).to(device),
            image_wh=torch.from_numpy(m['image_wh'][None]).to(device),
            timestamp=torch.tensor([m['timestamp']], dtype=torch.float64, device=device),
            img_metas=[dict(T_global=m['T_global'], T_global_inv=m['T_global_inv'])],
        )

    model.head.instance_bank.reset()
    if hasattr(model.head, 'sampler'):
        model.head.sampler.dn_metas = None
    with torch.inference_mode():
        first = one(0)
        for _ in range(max(0, warmup)):
            model.simple_test(**first)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        model.head.instance_bank.reset()

    out_dir = prep / 'e2e_predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / 'predictions.jsonl'
    timings: List[float] = []
    with pred_path.open('w', encoding='utf-8') as fout, torch.inference_mode():
        for i, info in enumerate(infos):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            data = one(i)
            native = model.simple_test(**data)[0]['img_bbox']
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
            boxes = native['boxes_3d']
            boxes = getattr(boxes, 'tensor', boxes)
            ids = native.get('instance_ids')
            rec = export_boxes(
                boxes.detach().cpu().numpy(),
                native['scores_3d'].detach().cpu().numpy(),
                native['labels_3d'].detach().cpu().numpy(),
                info,
                'sparse4d',
                None if ids is None else ids.detach().cpu().numpy(),
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

    arr = np.asarray(timings, dtype=np.float64)
    return {
        'predictions': str(pred_path),
        'poses': str(converted / 'poses_10hz.json'),
        'gt_common': str(converted / 'gt_common.jsonl'),
        'checkpoint': str(ckpt),
        'default_time_interval_s': DEFAULT_DT,
        'detector_hz': 10,
        'n_frames': len(timings),
        'sparse4d_latency_s': {
            'mean': float(arr.mean()),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'sum': float(arr.sum()),
        },
        'load_info': {'missing_keys': list(missing), 'unexpected_keys': list(unexpected)},
        'use_deformable_func': False,
        'timings_s': timings,
    }


def eval_det_ap(gt_path: Path, pred_path: Path) -> Dict[str, Any]:
    ensure_sparse4d_paths()
    from evaluate import evaluate

    gt = [json.loads(x) for x in gt_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    pr = [json.loads(x) for x in pred_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    return evaluate(gt, pr)


def run_hivt(
    episode: str,
    pred_path: Path,
    poses_path: Path,
    ckpt: Path,
    device: str,
    sparse4d_latency_s: float,
    max_windows: int = 10,
    gt_common_path: Path = None,
    history_source: str = 'sparse4d',
) -> Dict[str, Any]:
    ensure_import_paths()
    from integration.av2_adapter import world_predictions_av2
    from integration.bridge import iter_av2_windows, iter_av2_windows_gt
    from integration.forecast_metrics import load_av_poses, load_gt_tracks, min_ade_fde_mr
    from integration.risk_fusion import (
        build_risk_packet,
        detections_from_hivt_window,
        summarize_aligned_packet,
    )
    from integration.time_sync import sync_forecast
    from integration.run_e2e_smoke import arrays_to_temporal
    from models.hivt import HiVT

    diagnostics: Dict[str, Any] = {}
    poses_rows = json.loads(Path(poses_path).read_text(encoding='utf-8'))
    if history_source == 'gt':
        if gt_common_path is None:
            raise ValueError('gt history_source requires gt_common_path')
        windows = list(
            iter_av2_windows_gt(
                episode, gt_common_path, poses_path,
                max_windows=max_windows, diagnostics=diagnostics,
            )
        )
    else:
        windows = list(
            iter_av2_windows(
                episode, pred_path, poses_path, max_windows=max_windows, diagnostics=diagnostics
            )
        )
    model = HiVT.load_from_checkpoint(checkpoint_path=str(ckpt))
    model.enable_fast_inference()
    model.eval().to(device)
    assert getattr(model.hparams, 'use_actor_type', False), 'expected class-aware HiVT'

    if not windows:
        return {'ok': False, 'diagnostics': diagnostics, 'cases': []}

    gt_tracks = load_gt_tracks(str(gt_common_path)) if gt_common_path else {}
    av_track = load_av_poses(str(poses_path))

    # warmup
    warm = arrays_to_temporal(windows[0]).to(device)
    warm.y = None
    with torch.no_grad():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        _ = model(warm)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()

    cases = []
    pipe_latencies = []
    focal_rows, all_rows = [], []

    def _pack(fc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'n_agents': fc.get('n_agents'),
            'minADE': fc.get('minADE'),
            'minFDE': fc.get('minFDE'),
            'MR': fc.get('MR'),
            'ADE@30': fc.get('ADE@30'),
            'FDE@30': fc.get('FDE@30'),
            'MR@30': fc.get('MR@30'),
            'n_ok': sum(1 for d in fc.get('details', []) if d.get('status') == 'ok'),
            'n_no_assoc': sum(1 for d in fc.get('details', []) if d.get('status') == 'no_gt_assoc'),
        }

    for wi, arrays in enumerate(windows):
        data = arrays_to_temporal(arrays).to(device)
        data.y = None
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            with torch.no_grad():
                y_hat, pi = model(data)
            ender.record()
            torch.cuda.synchronize()
            hivt_s = starter.elapsed_time(ender) / 1000.0
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                y_hat, pi = model(data)
            hivt_s = time.perf_counter() - t0

        y_np = y_hat.detach().cpu().numpy()
        pi_np = torch.softmax(pi, dim=-1).detach().cpu().numpy()
        world = world_predictions_av2(y_np, arrays)
        t_obs = float(arrays['end_timestamp'])
        e2e_s = float(sparse4d_latency_s) + float(hivt_s)
        pipe_latencies.append(e2e_s)
        synced = sync_forecast(
            observation_time=t_obs,
            world_xy=world,
            mode_probabilities=pi_np,
            sparse4d_latency_s=sparse4d_latency_s,
            hivt_latency_s=hivt_s,
        )
        s4d_objects = detections_from_hivt_window(arrays)
        risk = build_risk_packet(
            t_obs=t_obs,
            sparse4d_latency_s=sparse4d_latency_s,
            hivt_latency_s=hivt_s,
            b2d_payload={'episode_id': episode},
            sparse4d_payload={
                'n_tracks': int(arrays['num_nodes']),
                'objects': s4d_objects,
            },
            hivt_world_xy=world,
            hivt_mode_probabilities=pi_np,
            poses=poses_rows,
        )
        # AV2 val protocol = focal agent only; also keep all-agent average for diagnostics.
        fc_focal = min_ade_fde_mr(world, arrays, gt_tracks, av_track, focal_only=True)
        fc_all = min_ade_fde_mr(world, arrays, gt_tracks, av_track, focal_only=False)
        if fc_focal.get('minADE') is not None:
            focal_rows.append(fc_focal)
        if fc_all.get('minADE') is not None:
            all_rows.append(fc_all)
        cases.append(
            {
                'window': wi,
                't_obs': t_obs,
                'num_nodes': int(arrays['num_nodes']),
                'actor_type_unique': sorted(set(int(x) for x in arrays['actor_type'].tolist())),
                'map_n_lane_vectors': int(arrays.get('map_n_lane_vectors', arrays['lane_vectors'].shape[0])),
                'map_n_lane_actor_edges': int(arrays.get('map_n_lane_actor_edges', arrays['lane_actor_index'].shape[1])),
                'map_n_intersection_vecs': int(arrays.get('map_n_intersection_vecs', int(arrays['is_intersections'].sum()))),
                'map_source': arrays.get('map_source', 'empty'),
                'hivt_latency_s': hivt_s,
                'e2e_latency_s_s4d_p50_plus_hivt': e2e_s,
                'finite_ok': bool(np.isfinite(world).all() and np.isfinite(pi_np).all()),
                'time_sync_skip': synced['time_sync']['skipped_future_steps'],
                't_fuse': risk['t_fuse'],
                'risk_aligned': summarize_aligned_packet(risk),
                'forecast_focal': _pack(fc_focal),
                'forecast_all': _pack(fc_all),
            }
        )

    def _agg(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        if not rows:
            return {'n_windows_with_agents': 0, 'minADE': None, 'minFDE': None, 'MR': None}
        return {
            'protocol': label,
            'n_windows_with_agents': len(rows),
            'minADE': float(np.mean([r['minADE'] for r in rows])),
            'minFDE': float(np.mean([r['minFDE'] for r in rows])),
            'MR': float(np.mean([r['MR'] for r in rows])),
            'ADE@30': float(np.mean([r['ADE@30'] for r in rows if r.get('ADE@30') is not None])) if any(r.get('ADE@30') is not None for r in rows) else None,
            'FDE@30': float(np.mean([r['FDE@30'] for r in rows if r.get('FDE@30') is not None])) if any(r.get('FDE@30') is not None for r in rows) else None,
            'MR@30': float(np.mean([r['MR@30'] for r in rows if r.get('MR@30') is not None])) if any(r.get('MR@30') is not None for r in rows) else None,
        }

    arr = np.asarray(pipe_latencies, dtype=np.float64)
    forecast_focal = _agg(
        focal_rows,
        'AV2-like focal agent only; GT gt_common interp; T_obs nearest assoc 3m; ADE@60 default',
    )
    forecast_all = _agg(
        all_rows,
        'all eligible agents (incl. AV); same GT association',
    )
    return {
        'ok': all(c['finite_ok'] for c in cases),
        'history_source': history_source,
        'hist_fut': [model.historical_steps, model.future_steps],
        'use_actor_type': True,
        'diagnostics': diagnostics,
        'n_windows': len(cases),
        'hivt_latency_s': {
            'mean': float(np.mean([c['hivt_latency_s'] for c in cases])),
            'p50': float(np.percentile([c['hivt_latency_s'] for c in cases], 50)),
        },
        'e2e_latency_s': {
            'mean': float(arr.mean()) if len(arr) else None,
            'p50': float(np.percentile(arr, 50)) if len(arr) else None,
            'p95': float(np.percentile(arr, 95)) if len(arr) else None,
            'note': 'per window: Sparse4D_p50 + HiVT_forward (native 10Hz detections, no temporal densify)',
        },
        'forecast_metrics': forecast_focal,
        'forecast_metrics_all_agents': forecast_all,
        'cases': cases,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', default=DEFAULT_EP)
    p.add_argument('--prepared', type=Path, default=DEFAULT_PREP)
    p.add_argument(
        '--ckpt',
        type=Path,
        default=DEFAULT_HIVT_CKPT,
    )
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max_windows', type=int, default=10)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--skip_sparse4d', action='store_true', help='reuse existing e2e_predictions')
    p.add_argument(
        '--history_source', choices=('sparse4d', 'gt', 'both'), default='both',
        help='sparse4d=E2E detections; gt=oracle GT tracks; both=compare',
    )
    args = p.parse_args()

    out = args.output or (OUTPUTS_ROOT / 'b2d_e2e')
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'

    print('========== B2D E2E (HiVT env) ==========')
    print('prepared', args.prepared)
    print('device', device)

    cache_complete = args.prepared / 'prepared' / 'cache' / args.episode / 'complete.json'
    if not cache_complete.is_file():
        raise FileNotFoundError(
            f'Missing RGB cache: {cache_complete}. Run prepare/cache in HiVT env first.'
        )

    if args.skip_sparse4d:
        s4d = {
            'predictions': str(args.prepared / 'e2e_predictions' / 'predictions.jsonl'),
            'poses': str(args.prepared / 'converted' / args.episode / 'poses_10hz.json'),
            'gt_common': str(args.prepared / 'converted' / args.episode / 'gt_common.jsonl'),
            'sparse4d_latency_s': {'p50': 0.129},
            'n_frames': None,
            'skipped': True,
        }
    else:
        print('--- Sparse4D ---')
        s4d = run_sparse4d(args.prepared, args.episode, device)
        print(json.dumps(s4d['sparse4d_latency_s'], indent=2))

    print('--- Detection AP ---')
    try:
        det = eval_det_ap(Path(s4d['gt_common']), Path(s4d['predictions']))
    except Exception as e:
        det = {'error': repr(e)}
    print(json.dumps({k: det[k] for k in det if k != 'matches'} if isinstance(det, dict) else det, indent=2, default=str)[:2000])

    sources = ['sparse4d', 'gt'] if args.history_source == 'both' else [args.history_source]
    hivt_by_src: Dict[str, Any] = {}
    for src in sources:
        print(f'--- HiVT history={src} ---')
        hivt = run_hivt(
            args.episode,
            Path(s4d['predictions']),
            Path(s4d['poses']),
            args.ckpt,
            device,
            sparse4d_latency_s=float(s4d['sparse4d_latency_s']['p50']),
            max_windows=args.max_windows,
            gt_common_path=Path(s4d['gt_common']),
            history_source=src,
        )
        hivt_by_src[src] = hivt
        print('ok', hivt.get('ok'), 'windows', hivt.get('n_windows'))
        print('latency', json.dumps(hivt.get('e2e_latency_s'), indent=2))
        print('forecast_focal', json.dumps(hivt.get('forecast_metrics'), indent=2))
        print('forecast_all', json.dumps(hivt.get('forecast_metrics_all_agents'), indent=2))

    hivt = hivt_by_src.get('sparse4d') or next(iter(hivt_by_src.values()))
    report = {
        'ok': bool(hivt.get('ok')) and 'error' not in det,
        'env_policy': 'HiVT conda only',
        'episode': args.episode,
        'baseline_note': (
            'AV2 val minADE≈0.95 uses GT tracks + map on Argoverse2, not B2D+Sparse4D. '
            'Compare hivt_by_history_source.gt (oracle) vs .sparse4d (E2E).'
        ),
        'sparse4d': {k: v for k, v in s4d.items() if k != 'timings_s'},
        'detection': det,
        'hivt': hivt,
        'hivt_by_history_source': hivt_by_src,
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print('wrote', out / 'report.json')
    if not report['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
