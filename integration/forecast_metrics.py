"""HiVT forecast ADE/FDE against Bench2Drive GT (gt_common.jsonl).

Predicted track_ids from Sparse4D/tracker usually do not equal CARLA GT ids, so
agents are associated by nearest GT center at T_obs (class-aware), then GT is
sampled along that track for future steps.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 — Tuple used by horizons / sample_xy

import numpy as np

from integration.paths import ensure_import_paths

ensure_import_paths()

from datasets.av2_constants import AV2_FUTURE_STEPS


def load_gt_tracks(gt_jsonl: str, dynamic_classes: Optional[set] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """Return track_id -> {'t': [T], 'xy': [T,2], 'cls': str} from gt_common (native Hz)."""
    import json
    from pathlib import Path

    if dynamic_classes is None:
        dynamic_classes = {
            'car', 'van', 'truck', 'bus', 'pedestrian', 'walker', 'motorcycle', 'bicycle', 'cyclist',
        }
    by_tid: Dict[str, List[Tuple[float, np.ndarray, str]]] = defaultdict(list)
    for line in Path(gt_jsonl).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        t = float(row['timestamp'])
        for o in row['objects']:
            if not o.get('valid_mask', True):
                continue
            cls = o.get('class')
            if cls not in dynamic_classes:
                continue
            tid = str(o['track_id'])
            xy = np.asarray(o['center_xyz'][:2], dtype=np.float64)
            by_tid[tid].append((t, xy, cls))

    out = {}
    for tid, rows in by_tid.items():
        rows = sorted(rows, key=lambda x: x[0])
        out[tid] = {
            't': np.asarray([r[0] for r in rows], dtype=np.float64),
            'xy': np.stack([r[1] for r in rows], axis=0),
            'cls': rows[-1][2],
        }
    return out


def load_av_poses(poses_json: str) -> Dict[str, np.ndarray]:
    import json
    from pathlib import Path

    poses = json.loads(Path(poses_json).read_text(encoding='utf-8'))
    t = np.asarray([float(p['timestamp']) for p in poses], dtype=np.float64)
    xy = np.stack(
        [np.asarray(p['ego_to_world'], dtype=np.float64)[:2, 3] for p in poses],
        axis=0,
    )
    return {'t': t, 'xy': xy, 'cls': 'car'}


def sample_xy(track: Dict[str, np.ndarray], times: np.ndarray, max_gap: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolate track onto `times`. valid[k]=False if outside coverage or gap>max_gap."""
    t = track['t']
    xy = track['xy']
    out = np.full((len(times), 2), np.nan, dtype=np.float64)
    valid = np.zeros(len(times), dtype=bool)
    if len(t) == 0:
        return out, valid
    for i, tt in enumerate(times):
        if tt < t[0] - 1e-9 or tt > t[-1] + 1e-9:
            continue
        j = int(np.searchsorted(t, tt))
        if j == 0:
            if abs(t[0] - tt) <= 1e-6:
                out[i] = xy[0]
                valid[i] = True
            continue
        if j >= len(t):
            if abs(t[-1] - tt) <= 1e-6:
                out[i] = xy[-1]
                valid[i] = True
            continue
        t0, t1 = t[j - 1], t[j]
        if t1 - t0 > max_gap + 1e-9:
            continue
        if abs(t1 - t0) < 1e-12:
            out[i] = xy[j]
        else:
            w = (tt - t0) / (t1 - t0)
            out[i] = (1 - w) * xy[j - 1] + w * xy[j]
        valid[i] = True
    return out, valid


def associate_agent_to_gt(
    agent_xy: np.ndarray,
    agent_cls: str,
    t_obs: float,
    gt_tracks: Dict[str, Dict[str, np.ndarray]],
    max_dist: float = 3.0,
) -> Optional[str]:
    """Nearest GT track of compatible class at t_obs."""
    from integration.av2_adapter import CLASS_TO_ACTOR_TYPE

    agent_type = CLASS_TO_ACTOR_TYPE.get(agent_cls, 5)
    best_tid, best_d = None, max_dist
    for tid, tr in gt_tracks.items():
        gt_type = CLASS_TO_ACTOR_TYPE.get(tr['cls'], 5)
        if agent_type != gt_type and not (agent_type == 0 and gt_type == 0):
            continue
        xy, valid = sample_xy(tr, np.asarray([t_obs]))
        if not valid[0]:
            continue
        d = float(np.linalg.norm(xy[0] - agent_xy))
        if d < best_d:
            best_d, best_tid = d, tid
    return best_tid


def _horizon_metrics(
    world_xy: np.ndarray,
    agent_i: int,
    gt_xy: np.ndarray,
    gt_valid: np.ndarray,
    horizons: Tuple[int, ...],
    mr_threshold: float,
) -> Dict[str, Any]:
    """Per-agent minADE/minFDE/MR at one or more future horizons (in steps)."""
    out = {}
    for h in horizons:
        h_use = min(h, world_xy.shape[2], gt_xy.shape[0])
        valid = gt_valid[:h_use]
        if valid.sum() < max(3, h_use // 4):
            out[f'ADE@{h}'] = None
            out[f'FDE@{h}'] = None
            out[f'MR@{h}'] = None
            continue
        pred = world_xy[:, agent_i, :h_use, :]
        err = np.linalg.norm(pred - gt_xy[None, :h_use, :], axis=-1)
        err = np.where(valid[None, :], err, np.nan)
        ade = np.nanmean(err, axis=-1)
        last = int(np.where(valid)[0][-1])
        fde = err[:, last]
        best = int(np.nanargmin(ade))
        out[f'ADE@{h}'] = float(ade[best])
        out[f'FDE@{h}'] = float(fde[best])
        out[f'MR@{h}'] = float(fde[best] > mr_threshold)
        if h == horizons[-1]:
            out['best_mode'] = best
            out['minADE'] = float(ade[best])
            out['minFDE'] = float(fde[best])
            out['miss'] = bool(fde[best] > mr_threshold)
    return out


def min_ade_fde_mr(
    world_xy: np.ndarray,
    arrays: Dict[str, Any],
    gt_tracks: Dict[str, Dict[str, np.ndarray]],
    av_track: Dict[str, np.ndarray],
    dt: float = 0.1,
    future_steps: int = AV2_FUTURE_STEPS,
    mr_threshold: float = 2.0,
    assoc_dist: float = 3.0,
    focal_only: bool = False,
    horizons: Tuple[int, ...] = (30, 60),
) -> Dict[str, Any]:
    """Compute minADE / minFDE / MR over modes for agents with GT association.

    world_xy: [F, N, H, 2] world-frame HiVT predictions

    By default matches AV2 HiVT val protocol when focal_only=True (agent_index only).
    Also reports ADE@30 (3s) and ADE@60 (6s).
    """
    if world_xy.ndim != 4 or world_xy.shape[-1] != 2:
        raise ValueError(f'world_xy expected [F,N,H,2], got {world_xy.shape}')
    f, n, h, _ = world_xy.shape
    h_use = min(h, future_steps)
    horizons = tuple(hh for hh in horizons if hh <= h_use) or (h_use,)
    t_obs = float(arrays['end_timestamp'])
    fut_times = t_obs + (np.arange(h_use) + 1) * dt
    track_ids = list(arrays['track_ids'])
    actor_type = np.asarray(arrays['actor_type'])
    eligible = np.asarray(arrays['prediction_eligible'], dtype=bool)
    type_to_name = {0: 'car', 1: 'pedestrian', 2: 'motorcycle', 3: 'bicycle', 4: 'bus', 5: 'car'}

    from datasets.av2_constants import AV2_HISTORICAL_STEPS

    positions_local = arrays['positions'][:, AV2_HISTORICAL_STEPS - 1, :]
    theta = float(arrays['theta'])
    origin = np.asarray(arrays['origin']).reshape(2)
    c, s = np.cos(theta), np.sin(theta)
    scene = np.array([[c, -s], [s, c]])
    cur_world = positions_local @ scene.T + origin

    agent_index = int(arrays['agent_index'])
    ades, fdes, mrs = [], [], []
    ade30, fde30, mr30 = [], [], []
    details = []
    for i in range(n):
        if not eligible[i]:
            continue
        if focal_only and i != agent_index:
            continue
        tid = str(track_ids[i])
        cls_name = type_to_name.get(int(actor_type[i]), 'car')
        is_av = (':AV' in tid)
        if is_av:
            # AV2 val metrics are on the focal predicted agent, not ego-as-AV filler.
            if focal_only:
                continue
            gt_tid = '__AV__'
            gt = av_track
            assoc_d = 0.0
        else:
            gt_tid = associate_agent_to_gt(cur_world[i], cls_name, t_obs, gt_tracks, max_dist=assoc_dist)
            if gt_tid is None:
                details.append({'agent': i, 'track_id': tid, 'status': 'no_gt_assoc', 'focal': i == agent_index})
                continue
            gt = gt_tracks[gt_tid]
            assoc_d = float(np.linalg.norm(sample_xy(gt, np.asarray([t_obs]))[0][0] - cur_world[i]))

        gt_xy, gt_valid = sample_xy(gt, fut_times)
        if gt_valid.sum() < max(5, h_use // 4):
            details.append({
                'agent': i, 'track_id': tid, 'gt_tid': gt_tid,
                'status': 'insufficient_gt_future', 'n_valid': int(gt_valid.sum()),
                'focal': i == agent_index,
            })
            continue

        hm = _horizon_metrics(world_xy, i, gt_xy, gt_valid, horizons, mr_threshold)
        if hm.get('minADE') is None:
            details.append({'agent': i, 'track_id': tid, 'status': 'horizon_invalid', 'focal': i == agent_index})
            continue
        ades.append(hm['minADE'])
        fdes.append(hm['minFDE'])
        mrs.append(float(hm['miss']))
        if hm.get('ADE@30') is not None:
            ade30.append(hm['ADE@30'])
            fde30.append(hm['FDE@30'])
            mr30.append(hm['MR@30'])
        details.append({
            'agent': i,
            'track_id': tid,
            'gt_tid': gt_tid,
            'class': cls_name,
            'assoc_dist_m': assoc_d,
            'n_gt_future': int(gt_valid.sum()),
            'focal': i == agent_index,
            'best_mode': hm.get('best_mode'),
            'minADE': hm['minADE'],
            'minFDE': hm['minFDE'],
            'miss': hm['miss'],
            'ADE@30': hm.get('ADE@30'),
            'FDE@30': hm.get('FDE@30'),
            'ADE@60': hm.get('ADE@60', hm['minADE']),
            'FDE@60': hm.get('FDE@60', hm['minFDE']),
            'status': 'ok',
        })

    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    if not ades:
        return {
            'n_agents': 0,
            'minADE': None,
            'minFDE': None,
            'MR': None,
            'ADE@30': None,
            'FDE@30': None,
            'MR@30': None,
            'focal_only': focal_only,
            'details': details,
        }
    return {
        'n_agents': len(ades),
        'minADE': _mean(ades),
        'minFDE': _mean(fdes),
        'MR': _mean(mrs),
        'ADE@30': _mean(ade30),
        'FDE@30': _mean(fde30),
        'MR@30': _mean(mr30),
        'focal_only': focal_only,
        'details': details,
    }
