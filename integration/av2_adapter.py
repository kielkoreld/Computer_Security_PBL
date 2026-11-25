"""Build AV2-HiVT TemporalData fields from dense world-frame track histories."""
from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Optional, Sequence

import numpy as np

from integration.paths import ensure_import_paths

ensure_import_paths()

from datasets.av2_constants import (
    AV2_FUTURE_STEPS,
    AV2_HISTORICAL_STEPS,
    AV2_TOTAL_STEPS,
)

# Sparse4D / Bench2Drive class string → HiVT actor_type id
CLASS_TO_ACTOR_TYPE = {
    'car': 0,
    'van': 0,
    'truck': 0,
    'bus': 4,
    'pedestrian': 1,
    'walker': 1,
    'motorcycle': 2,
    'bicycle': 3,
    'cyclist': 3,
}
DYNAMIC_CLASSES = set(CLASS_TO_ACTOR_TYPE)


def empty_av2_map() -> Dict[str, np.ndarray]:
    return {
        'lane_vectors': np.zeros((0, 2), dtype=np.float32),
        'is_intersections': np.zeros((0,), dtype=np.uint8),
        'lane_types': np.zeros((0,), dtype=np.uint8),
        'left_mark_types': np.zeros((0,), dtype=np.uint8),
        'right_mark_types': np.zeros((0,), dtype=np.uint8),
        'lane_actor_index': np.empty((2, 0), dtype=np.int64),
        'lane_actor_vectors': np.zeros((0, 2), dtype=np.float32),
    }


def history_tensors_av2(
    records: Sequence[dict],
    episode_id: str,
    end_timestamp: float,
    av_track_id: str,
    agent_track_id: str,
    track_classes: Optional[Dict[str, str]] = None,
    dt: float = 0.1,
) -> Dict:
    """AV2 contract: hist=50, future slots=60, actor_type, empty map allowed for smoke."""
    if abs(dt - 0.1) > 1e-9:
        raise ValueError('AV2 HiVT expects dt=0.1s')
    hist = AV2_HISTORICAL_STEPS
    total = AV2_TOTAL_STEPS
    t0 = end_timestamp - (hist - 1) * dt
    records = [
        r for r in records
        if r['timestamp'] <= end_timestamp + 1e-7 and r['timestamp'] >= t0 - 1e-7
    ]
    if not records:
        raise ValueError('이력 없음')
    if any(r['episode_id'] != episode_id for r in records):
        raise ValueError('episode 혼합')
    if any(r['coordinate_frame'] != 'world_rh' for r in records):
        raise ValueError('world_rh로 변환 필요')

    ids = sorted({r['track_id'] for r in records})
    if av_track_id not in ids or agent_track_id not in ids:
        raise ValueError('AV/AGENT 누락')
    index = {x: i for i, x in enumerate(ids)}
    n = len(ids)

    positions = np.zeros((n, total, 2), dtype=np.float32)
    mask = np.ones((n, total), dtype=bool)
    seen = set()
    class_of = {tid: 'unknown' for tid in ids}
    if track_classes:
        for tid, c in track_classes.items():
            if tid in class_of:
                class_of[tid] = c

    for r in records:
        t = (r['timestamp'] - t0) / dt
        step = int(round(t))
        if abs(t - step) > 1e-5 or not 0 <= step < hist:
            raise ValueError('0.1초 격자 불일치')
        i = index[r['track_id']]
        if (i, step) in seen:
            raise ValueError('중복 ID/시각')
        seen.add((i, step))
        if r.get('class'):
            class_of[r['track_id']] = r['class']
        if r.get('valid_mask', True):
            xy = np.asarray(r['center_xyz'], dtype=np.float64)[:2]
            if not np.isfinite(xy).all():
                raise ValueError('좌표 NaN')
            positions[i, step] = xy
            mask[i, step] = False

    av = index[av_track_id]
    if mask[av, :hist].any():
        raise ValueError(f'{hist}개 AV pose 필요')
    origin = positions[av, hist - 1].astype(np.float64)
    heading = positions[av, hist - 1] - positions[av, hist - 2]
    if np.linalg.norm(heading) < 1e-4:
        raise ValueError('정지 AV의 이동 heading 불명확')
    theta = float(np.arctan2(heading[1], heading[0]))
    c, s = np.cos(theta), np.sin(theta)
    rotate = np.array([[c, -s], [s, c]], dtype=np.float64)

    for i in range(n):
        valid = ~mask[i, :hist]
        positions[i, :hist][valid] = (positions[i, :hist][valid] - origin) @ rotate

    # Actors invisible at current step: no future (dataset convention).
    for i in range(n):
        if mask[i, hist - 1]:
            mask[i, hist:] = True

    x = np.zeros((n, hist, 2), dtype=np.float32)
    valid_pair = ~(mask[:, : hist - 1] | mask[:, 1:hist])
    x[:, 1:] = np.where(
        valid_pair[..., None],
        positions[:, 1:hist] - positions[:, : hist - 1],
        0.0,
    )

    bos = np.zeros((n, hist), dtype=bool)
    bos[:, 0] = ~mask[:, 0]
    bos[:, 1:] = mask[:, : hist - 1] & ~mask[:, 1:hist]

    angles = np.zeros(n, dtype=np.float32)
    eligible = np.zeros(n, dtype=bool)
    actor_type = np.full(n, 5, dtype=np.int64)  # UNKNOWN default
    for i, tid in enumerate(ids):
        valid = np.where(~mask[i, :hist])[0]
        if len(valid) > 1:
            vec = positions[i, valid[-1]] - positions[i, valid[-2]]
            angles[i] = np.arctan2(vec[1], vec[0])
            eligible[i] = not mask[i, hist - 1]
        else:
            mask[i, hist:] = True
        actor_type[i] = CLASS_TO_ACTOR_TYPE.get(class_of[tid], 5)
    # AV is treated as vehicle.
    actor_type[av] = 0

    if n >= 2:
        edges = np.asarray(list(permutations(range(n), 2)), dtype=np.int64).T
    else:
        edges = np.empty((2, 0), dtype=np.int64)

    out = {
        'x': x,
        'positions': positions.astype(np.float32),
        'padding_mask': mask,
        'bos_mask': bos,
        'rotate_angles': angles,
        'edge_index': edges,
        'num_nodes': n,
        'av_index': av,
        'agent_index': index[agent_track_id],
        'origin': origin.astype(np.float32)[None],
        'theta': theta,
        'y': None,
        'actor_type': actor_type,
        'track_ids': ids,
        'prediction_eligible': eligible,
        'episode_id': episode_id,
        'end_timestamp': float(end_timestamp),
        'city': 'bench2drive',
        'seq_id': episode_id,
        **empty_av2_map(),
    }
    return out


def world_predictions_av2(pred: np.ndarray, arrays: Dict) -> np.ndarray:
    """pred [F,N,H,2|4] in agent-local deltas → world XY [F,N,H,2].

    Time base: anchored at history end ``positions[:, hist-1]`` (= T_obs /
    ``end_timestamp``). Step k is the pose at T_obs + (k + 1) * 0.1 s — not
    relative to HiVT wall-clock finish time.
    """
    hist = AV2_HISTORICAL_STEPS
    angles = arrays['rotate_angles']
    c = np.cos(angles)
    s = np.sin(angles)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    local = np.einsum('fnti,nji->fntj', pred[..., :2], rot) + arrays['positions'][None, :, hist - 1:hist, :]
    theta = arrays['theta']
    scene = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    world = local @ scene.T + arrays['origin']
    if not np.isfinite(world).all():
        raise ValueError('world prediction NaN')
    return world
