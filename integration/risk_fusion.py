"""Risk-judgment timeline alignment for Bench2Drive / Sparse4D / HiVT.

Output time bases (verified in adapters/models — not wall-clock ready time):
  - Sparse4D boxes describe geometry at the camera frame T_obs.
  - HiVT futures are offsets from history end (positions[:, hist-1] @ T_obs);
    absolute step k is T_obs + (k + 1) * dt. They are NOT relative to inference end.
  - Bench2Drive poses are stamped at their own timestamps.

Fusion clock (consumer "now"):
  - Default T_fuse = T_hivt_ready = T_obs + L_s4d + L_hivt
    (when the HiVT result is available, after Sparse4D then HiVT).

Projection onto that single clock:
  - HiVT: keep future steps with t_k >= T_fuse (skip earlier)
  - Sparse4D: CV-propagate centers T_obs → T_fuse
  - Bench2Drive: interpolate ego pose to T_fuse
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from integration.paths import ensure_import_paths

ensure_import_paths()

from datasets.av2_constants import AV2_HISTORICAL_STEPS


@dataclass
class StampedSource:
    """One module's contribution with explicit time semantics."""
    source: str  # 'bench2drive' | 'sparse4d' | 'hivt'
    observation_time: float
    available_time: float
    payload: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_xy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size < 2 or not np.isfinite(a[:2]).all():
        return None
    return a[:2].copy()


def _as_vel(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size < 2 or not np.isfinite(a[:2]).all():
        return None
    return a[:2].copy()


def sample_ego_pose(
    poses: Sequence[Dict[str, Any]],
    t: float,
) -> Dict[str, Any]:
    """Interpolate ego_to_world translation (and nearest rotation) at time t."""
    if not poses:
        raise ValueError('poses empty')
    ts = np.asarray([float(p['timestamp']) for p in poses], dtype=np.float64)
    order = np.argsort(ts)
    ts = ts[order]
    mats = [np.asarray(poses[i]['ego_to_world'], dtype=np.float64) for i in order]
    if t <= ts[0]:
        M = mats[0]
        return {
            'timestamp': float(ts[0]),
            'ego_xy': M[:2, 3].tolist(),
            'ego_to_world': M.tolist(),
            'method': 'clamp_start',
            'query_t': float(t),
        }
    if t >= ts[-1]:
        M = mats[-1]
        return {
            'timestamp': float(ts[-1]),
            'ego_xy': M[:2, 3].tolist(),
            'ego_to_world': M.tolist(),
            'method': 'clamp_end',
            'query_t': float(t),
        }
    j = int(np.searchsorted(ts, t))
    t0, t1 = ts[j - 1], ts[j]
    w = 0.0 if abs(t1 - t0) < 1e-12 else float((t - t0) / (t1 - t0))
    M0, M1 = mats[j - 1], mats[j]
    M = M0.copy()
    M[:3, 3] = (1.0 - w) * M0[:3, 3] + w * M1[:3, 3]
    # Keep rotation of the nearer pose (stable for short dt).
    M[:3, :3] = M0[:3, :3] if w < 0.5 else M1[:3, :3]
    return {
        'timestamp': float(t),
        'ego_xy': M[:2, 3].tolist(),
        'ego_to_world': M.tolist(),
        'method': 'linear_translation_interp',
        'query_t': float(t),
        'bracket': [float(t0), float(t1)],
    }


def detections_from_hivt_window(arrays: Dict[str, Any], dt: float = 0.1) -> List[Dict[str, Any]]:
    """Build T_obs-stamped detection dicts from an AV2 HiVT window (world XY + CV velocity)."""
    hist = AV2_HISTORICAL_STEPS
    mask = np.asarray(arrays['padding_mask'])
    origin = np.asarray(arrays['origin'], dtype=np.float64).reshape(2)
    theta = float(arrays['theta'])
    c, s = np.cos(theta), np.sin(theta)
    scene = np.array([[c, -s], [s, c]], dtype=np.float64)
    pos_local = np.asarray(arrays['positions'], dtype=np.float64)
    track_ids = list(arrays['track_ids'])
    actor_type = np.asarray(arrays['actor_type'])
    type_to_name = {0: 'car', 1: 'pedestrian', 2: 'motorcycle', 3: 'bicycle', 4: 'bus', 5: 'unknown'}
    out: List[Dict[str, Any]] = []
    for i, tid in enumerate(track_ids):
        if mask[i, hist - 1]:
            continue
        xy_l = pos_local[i, hist - 1, :2]
        xy_w = xy_l @ scene.T + origin
        vel = np.zeros(2, dtype=np.float64)
        if hist >= 2 and (not mask[i, hist - 2]):
            prev_w = pos_local[i, hist - 2, :2] @ scene.T + origin
            vel = (xy_w - prev_w) / dt
        out.append({
            'track_id': str(tid),
            'class': type_to_name.get(int(actor_type[i]), 'unknown'),
            'center_xyz': [float(xy_w[0]), float(xy_w[1]), 0.0],
            'velocity_xy': [float(vel[0]), float(vel[1])],
            'observation_time': float(arrays['end_timestamp']),
            'coordinate_frame': 'world_rh',
        })
    return out


def propagate_detections_to_t(
    objects: Sequence[Dict[str, Any]],
    t_obs: float,
    t_fuse: float,
) -> Dict[str, Any]:
    """Constant-velocity propagate detection centers from t_obs to t_fuse."""
    dt = float(t_fuse - t_obs)
    aligned = []
    n_with_v = 0
    n_no_v = 0
    for o in objects:
        xy = _as_xy(o.get('center_xyz') or o.get('center_xy'))
        if xy is None:
            continue
        vel = _as_vel(o.get('velocity_xy'))
        item = {
            'track_id': o.get('track_id'),
            'class': o.get('class'),
            'center_xy_at_obs': xy.tolist(),
            'center_xy_at_fuse': xy.tolist(),
            'velocity_xy': None,
            'propagated': False,
        }
        if vel is not None:
            item['velocity_xy'] = vel.tolist()
            item['center_xy_at_fuse'] = (xy + vel * dt).tolist()
            item['propagated'] = True
            n_with_v += 1
        else:
            n_no_v += 1
        aligned.append(item)
    return {
        'objects_at_fuse': aligned,
        'method': 'constant_velocity',
        'delta_t': dt,
        'n_objects': len(aligned),
        'n_propagated': n_with_v,
        'n_held_without_velocity': n_no_v,
        'applied': True,
    }


def align_hivt_to_t_fuse(
    hivt_world_xy: np.ndarray,
    hivt_mode_probabilities: np.ndarray,
    t_obs: float,
    t_fuse: float,
    dt: float = 0.1,
) -> Dict[str, Any]:
    if hivt_world_xy.ndim != 4 or hivt_world_xy.shape[-1] != 2:
        raise ValueError(f'hivt_world_xy expected [F,N,H,2], got {hivt_world_xy.shape}')
    h = hivt_world_xy.shape[2]
    abs_times = t_obs + (np.arange(h) + 1) * dt
    after = np.where(abs_times >= t_fuse - 1e-9)[0]
    idx0 = int(after[0]) if len(after) else h
    remain = h - idx0
    return {
        'position_at_fuse': (
            hivt_world_xy[:, :, idx0, :] if remain > 0 else None
        ),
        'future_world_xy': (
            hivt_world_xy[:, :, idx0:, :] if remain > 0 else hivt_world_xy[:, :, :0, :]
        ),
        'future_abs_times': abs_times[idx0:].tolist() if remain > 0 else [],
        'skipped_future_steps': idx0,
        'remaining_future_steps': remain,
        'mode_probabilities': hivt_mode_probabilities,
        'method': 'index_first_step_at_or_after_t_fuse',
        'applied': True,
    }


def build_risk_packet(
    t_obs: float,
    sparse4d_latency_s: float,
    hivt_latency_s: float,
    b2d_payload: Dict[str, Any],
    sparse4d_payload: Dict[str, Any],
    hivt_world_xy: np.ndarray,
    hivt_mode_probabilities: np.ndarray,
    dt: float = 0.1,
    t_fuse: Optional[float] = None,
    poses: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble a risk packet with all sources aligned onto T_fuse.

    Required for same-clock risk judgment:
      - HiVT futures indexed to T_fuse
      - Sparse4D boxes CV-propagated when velocity_xy is present
      - B2D ego sampled at T_fuse when ``poses`` (10Hz) is provided
    """
    l_s = float(sparse4d_latency_s)
    l_h = float(hivt_latency_s)
    t_s4d_ready = float(t_obs + l_s)
    t_hivt_ready = float(t_obs + l_s + l_h)
    # Joint risk clock = HiVT inference completion (pipeline ready), not T_obs.
    fusion_clock = 'hivt_ready'
    if t_fuse is None:
        t_fuse = t_hivt_ready
    else:
        fusion_clock = 'explicit_t_fuse'
    t_fuse = float(t_fuse)
    s4d_age = float(t_fuse - t_obs)

    module_time_bases = {
        'sparse4d': {
            'geometry_time': 'T_obs',
            'available_time': 'T_obs + L_s4d',
            'meaning': 'Detection centers are at the observed camera frame, not at ready time.',
        },
        'hivt': {
            'geometry_time': 'T_obs + (k+1)*dt for future step k',
            'available_time': 'T_obs + L_s4d + L_hivt',
            'meaning': (
                'Futures are conditioned on histories ending at T_obs '
                '(world = local_delta + positions[hist-1]); not rebased to inference end.'
            ),
        },
        'bench2drive': {
            'geometry_time': 'pose.timestamp',
            'available_time': 'pose.timestamp (sim state)',
            'meaning': 'Ego/world poses are sampled by absolute timestamp.',
        },
    }

    # --- HiVT ---
    hivt_aligned = align_hivt_to_t_fuse(
        hivt_world_xy, hivt_mode_probabilities, t_obs, t_fuse, dt=dt,
    )

    # --- Sparse4D ---
    objects = list(sparse4d_payload.get('objects') or [])
    s4d_aligned = propagate_detections_to_t(objects, t_obs, t_fuse) if objects else {
        'objects_at_fuse': [],
        'method': 'constant_velocity',
        'delta_t': s4d_age,
        'n_objects': 0,
        'n_propagated': 0,
        'n_held_without_velocity': 0,
        'applied': False,
        'reason': 'sparse4d_payload.objects missing',
    }

    # --- Bench2Drive ego ---
    b2d_at_obs = None
    b2d_at_fuse = None
    b2d_method = 'none'
    if poses is not None:
        b2d_at_obs = sample_ego_pose(poses, t_obs)
        b2d_at_fuse = sample_ego_pose(poses, t_fuse)
        b2d_method = b2d_at_fuse['method']
    elif 'ego_xy' in b2d_payload or 'ego_to_world' in b2d_payload:
        # Fallback: hold T_obs ego (explicitly marked as not advanced).
        xy = _as_xy(b2d_payload.get('ego_xy'))
        if xy is None and 'ego_to_world' in b2d_payload:
            M = np.asarray(b2d_payload['ego_to_world'], dtype=np.float64)
            xy = M[:2, 3]
        b2d_at_obs = {
            'timestamp': float(t_obs),
            'ego_xy': None if xy is None else xy.tolist(),
            'method': 'payload_at_obs',
        }
        b2d_at_fuse = {
            'timestamp': float(t_fuse),
            'ego_xy': None if xy is None else xy.tolist(),
            'method': 'held_at_obs_no_poses',
            'note': 'Provide poses=poses_10hz to interpolate ego to T_fuse',
        }
        b2d_method = 'held_at_obs_no_poses'

    aligned_at_t_fuse = {
        't_fuse': t_fuse,
        't_obs': t_obs,
        't_s4d_ready': t_s4d_ready,
        't_hivt_ready': t_hivt_ready,
        'fusion_clock': fusion_clock,
        'staleness_s': s4d_age,
        'delta_from_obs_s': s4d_age,
        'delta_from_s4d_ready_s': float(t_fuse - t_s4d_ready),
        'bench2drive': {
            'ego_at_fuse': b2d_at_fuse,
            'ego_at_obs': b2d_at_obs,
            'aligned': b2d_at_fuse is not None and b2d_method != 'held_at_obs_no_poses',
            'method': b2d_method,
            'correct_by': 'interp_poses_to_t_fuse',
        },
        'sparse4d': {
            **s4d_aligned,
            'aligned': bool(s4d_aligned.get('applied')),
            'correct_by': 'cv_propagate_obs_to_t_fuse',
        },
        'hivt': {
            'position_at_fuse': hivt_aligned['position_at_fuse'],
            'future_world_xy': hivt_aligned['future_world_xy'],
            'future_abs_times': hivt_aligned['future_abs_times'],
            'skipped_future_steps': hivt_aligned['skipped_future_steps'],
            'remaining_future_steps': hivt_aligned['remaining_future_steps'],
            'aligned': True,
            'method': hivt_aligned['method'],
            'correct_by': 'skip_steps_before_t_fuse',
        },
        'same_clock': bool(
            (b2d_at_fuse is not None and b2d_method != 'held_at_obs_no_poses')
            and bool(s4d_aligned.get('applied'))
            and True  # HiVT always indexed
        ),
    }

    sources = [
        StampedSource(
            source='bench2drive',
            observation_time=t_obs,
            available_time=t_obs,
            payload={
                **b2d_payload,
                'ego_at_obs': b2d_at_obs,
                'ego_at_fuse': b2d_at_fuse,
            },
            notes=[
                'Ego pose sampled at T_fuse from poses_10Hz when poses are provided.',
                'Do not mix a newer live B2D tick without updating Sparse4D/HiVT snapshot.',
            ],
        ).to_dict(),
        StampedSource(
            source='sparse4d',
            observation_time=t_obs,
            available_time=t_s4d_ready,
            payload={
                **{k: v for k, v in sparse4d_payload.items() if k != 'objects'},
                'n_objects_in': len(objects),
                'staleness_s': s4d_age,
                'objects_at_obs': objects,
                'aligned_to_fuse': s4d_aligned,
            },
            notes=[
                'Detections observed at T_obs; centers propagated to T_fuse with constant velocity when velocity_xy exists.',
            ],
        ).to_dict(),
        StampedSource(
            source='hivt',
            observation_time=t_obs,
            available_time=t_hivt_ready,
            payload={
                'mode_probabilities': hivt_mode_probabilities,
                **{k: hivt_aligned[k] for k in (
                    'position_at_fuse', 'future_world_xy', 'future_abs_times',
                    'skipped_future_steps', 'remaining_future_steps',
                )},
            },
            notes=[
                'Forecast conditioned on histories ending at T_obs; indexed onto T_fuse.',
            ],
        ).to_dict(),
    ]

    return {
        't_obs': t_obs,
        't_fuse': t_fuse,
        't_s4d_ready': t_s4d_ready,
        't_hivt_ready': t_hivt_ready,
        'fusion_clock': fusion_clock,
        'module_time_bases': module_time_bases,
        'latencies': {
            'sparse4d_s': l_s,
            'hivt_s': l_h,
            'total_s': l_s + l_h,
        },
        'aligned_at_t_fuse': aligned_at_t_fuse,
        'sources': sources,
        'fusion_rules': [
            'Sparse4D/HiVT geometry clocks are T_obs-based; ready times are later.',
            'Joint risk uses one consumer clock t_fuse (default = t_hivt_ready).',
            'Do not use per-module ready times as separate fuse clocks for the same judgment.',
            'Correction amount differs by module, but the target time is the same t_fuse.',
            'Bench2Drive: interpolate ego to t_fuse; Sparse4D: CV to t_fuse; HiVT: skip past steps.',
            'Use aligned_at_t_fuse for risk logic — all three domains on one clock.',
        ],
        'limitation': (
            'CV propagation assumes constant velocity over staleness (~L_s4d+L_hivt). '
            'Agents without velocity are held at T_obs position and flagged.'
        ),
    }


def summarize_aligned_packet(risk: Dict[str, Any]) -> Dict[str, Any]:
    """Compact JSON-safe summary for logs/reports (no large arrays)."""
    a = risk.get('aligned_at_t_fuse') or {}
    b2d = a.get('bench2drive') or {}
    s4d = a.get('sparse4d') or {}
    hivt = a.get('hivt') or {}
    ego = (b2d.get('ego_at_fuse') or {})
    return {
        't_obs': risk.get('t_obs'),
        't_fuse': risk.get('t_fuse'),
        't_hivt_ready': risk.get('t_hivt_ready'),
        'fusion_clock': risk.get('fusion_clock'),
        'latencies': risk.get('latencies'),
        'same_clock': a.get('same_clock'),
        'bench2drive': {
            'aligned': b2d.get('aligned'),
            'method': b2d.get('method'),
            'ego_xy_at_fuse': ego.get('ego_xy'),
            'correct_by': b2d.get('correct_by'),
        },
        'sparse4d': {
            'aligned': s4d.get('aligned'),
            'n_objects': s4d.get('n_objects'),
            'n_propagated': s4d.get('n_propagated'),
            'n_held_without_velocity': s4d.get('n_held_without_velocity'),
            'delta_t': s4d.get('delta_t'),
            'correct_by': s4d.get('correct_by'),
        },
        'hivt': {
            'aligned': hivt.get('aligned'),
            'skipped_future_steps': hivt.get('skipped_future_steps'),
            'remaining_future_steps': hivt.get('remaining_future_steps'),
            'correct_by': hivt.get('correct_by'),
        },
        'limitation': risk.get('limitation'),
    }
