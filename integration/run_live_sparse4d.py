"""Load Sparse4D from Sparse4D/checkpoints and run a real forward pass.

Uses handoff bundled 3-frame example inputs (same contract as
handoff/tools/sparse4d_example.py) and the 10Hz checkpoint
``checkpoints/sparse4dv3_r50_b2d_10hz_epoch14.pth``.

Outputs detections JSONL + measured Sparse4D latency for the HiVT bridge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from integration.paths import (
    DEFAULT_SPARSE4D_CKPT,
    OUTPUTS_ROOT,
    REPO_ROOT as REPO,
    SPARSE4D_ROOT,
    ensure_sparse4d_paths,
)

WEIGHT_CKPT = DEFAULT_SPARSE4D_CKPT
DEFAULT_DT = 0.1


def _resolve_checkpoint() -> Path:
    """Canonical 10Hz Sparse4D weight lives under Sparse4D/checkpoints/."""
    return WEIGHT_CKPT.resolve(strict=True)


def _ensure_anchor(ckpt_path: Path) -> Path:
    """Sparse4D configs need kmeans anchors dumped from the checkpoint."""
    import torch as _torch

    anchor_path = SPARSE4D_ROOT / 'handoff' / 'artifacts' / 'nuscenes_kmeans900_from_checkpoint.npy'
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    ck = _torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    anchor = sd['head.instance_bank.anchor'].detach().cpu().numpy()
    if anchor_path.exists():
        existing = np.load(anchor_path)
        if existing.shape != anchor.shape or not np.allclose(existing, anchor):
            # Prefer the live weight's anchors for this runner.
            np.save(anchor_path, anchor)
    else:
        np.save(anchor_path, anchor)
    return anchor_path


def _load_model(ckpt_path: Path, device: str):
    ensure_sparse4d_paths()
    from mmcv import Config
    from mmdet.models import build_detector
    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(
        str(SPARSE4D_ROOT / 'projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py')
    )
    cfg.model.img_backbone.pretrained = None
    cfg.model.head.instance_bank.anchor = str(
        SPARSE4D_ROOT / 'handoff/artifacts/nuscenes_kmeans900_from_checkpoint.npy'
    )
    cfg.model.head.instance_bank.default_time_interval = DEFAULT_DT
    # Without Sparse4D CUDA op (deformable_aggregation_ext), fall back to the
    # pure-PyTorch feature_sampling path. Same checkpoint; slightly different ops.
    def _disable_daf(obj) -> None:
        if isinstance(obj, dict):
            if 'use_deformable_func' in obj:
                obj['use_deformable_func'] = False
            for v in obj.values():
                _disable_daf(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _disable_daf(v)

    _disable_daf(cfg.model)
    model = build_detector(cfg.model).eval()
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    # Initial pretrained may not match every head buffer name exactly after
    # config tweaks; prefer strict when possible.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(device)
    return model, {'missing_keys': list(missing), 'unexpected_keys': list(unexpected)}


def _build_example_inputs(device: str):
    from model_inputs import export_boxes  # noqa: F401 — imported for side use below

    values = dict(np.load(SPARSE4D_ROOT / 'handoff/examples/sparse4d_3frames.npz'))
    infos = json.loads(
        (SPARSE4D_ROOT / 'handoff/examples/sparse4d_3frames_metadata.json').read_text()
    )['infos']
    lut = np.load(SPARSE4D_ROOT / 'handoff/cache/normalization_lut.npy')
    inputs = []
    for i in range(3):
        image = lut[np.arange(3)[None, :, None, None], values['rgb_uint8'][i]][None]
        data = dict(
            img=torch.from_numpy(image).to(device),
            projection_mat=torch.from_numpy(values['projection_mat'][i : i + 1]).to(device),
            image_wh=torch.from_numpy(values['image_wh'][i : i + 1]).to(device),
            timestamp=torch.from_numpy(values['timestamp'][i : i + 1]).to(device),
            img_metas=[
                dict(
                    T_global=values['T_global'][i],
                    T_global_inv=values['T_global_inv'][i],
                )
            ],
        )
        inputs.append(data)
    return inputs, infos, values


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--warmup', type=int, default=2)
    args = p.parse_args()

    out = args.output or (OUTPUTS_ROOT / 'live_sparse4d')
    out.mkdir(parents=True, exist_ok=True)

    ckpt = _resolve_checkpoint()
    _ensure_anchor(ckpt)
    device = args.device
    model, load_info = _load_model(ckpt, device)
    inputs, infos, values = _build_example_inputs(device)

    ensure_sparse4d_paths()
    from model_inputs import export_boxes

    # Warmup (temporal bank)
    model.head.instance_bank.reset()
    if hasattr(model.head, 'sampler') and hasattr(model.head.sampler, 'dn_metas'):
        model.head.sampler.dn_metas = None
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            model.simple_test(**inputs[0])
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    model.head.instance_bank.reset()
    if hasattr(model.head, 'sampler') and hasattr(model.head.sampler, 'dn_metas'):
        model.head.sampler.dn_metas = None

    rows = []
    latencies = []
    with torch.inference_mode():
        for data, info in zip(inputs, infos):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            native = model.simple_test(**data)[0]['img_bbox']
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

            boxes = native['boxes_3d']
            boxes = getattr(boxes, 'tensor', boxes)
            ids = native.get('instance_ids')
            rows.append(
                export_boxes(
                    boxes.detach().cpu().numpy(),
                    native['scores_3d'].detach().cpu().numpy(),
                    native['labels_3d'].detach().cpu().numpy(),
                    info,
                    'sparse4d',
                    None if ids is None else ids.detach().cpu().numpy(),
                )
            )

    pred_path = out / 'predictions.json'
    pred_path.write_text(json.dumps(rows, ensure_ascii=False, allow_nan=False))

    # One frame per line: {timestamp, objects:[...]} — same contract as bridge JSONL
    jsonl_path = out / 'detections.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as f:
        for frame in rows:
            f.write(json.dumps(frame, ensure_ascii=False) + '\n')

    summary = {
        'checkpoint': str(ckpt.resolve()),
        'weight_source': str(ckpt.resolve()),
        'device': device,
        'load_info': load_info,
        'n_frames': len(rows),
        'timestamps_s': values['timestamp'].tolist(),
        'latency_s': latencies,
        'latency_mean_s': float(np.mean(latencies)) if latencies else None,
        'predictions': str(pred_path),
        'detections_jsonl': str(jsonl_path),
        'note': (
            'Live Sparse4D forward with 10Hz epoch14 weight on handoff 3-frame example. '
            'use_deformable_func=False (PyTorch sampling) until deformable_aggregation_ext '
            'is built via handoff/tools/setup_sparse4d_cuda.sh. '
            'Use latency_mean_s as sparse4d_latency_s in risk_fusion / time_sync.'
        ),
        'use_deformable_func': False,
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
