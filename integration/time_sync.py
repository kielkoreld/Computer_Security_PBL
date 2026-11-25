"""Legacy HiVT-only latency trim + pointer to risk-fusion sync.

For risk judgment that must jointly use Bench2Drive + Sparse4D + HiVT, prefer
``integration.risk_fusion.build_risk_packet``. Dropping elapsed HiVT future steps
alone does **not** put Sparse4D boxes / B2D ego onto the same geometry clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from integration.paths import ensure_import_paths

ensure_import_paths()

from datasets.av2_constants import AV2_FUTURE_STEPS


@dataclass
class TimeSyncResult:
    """Latency-compensated view of a HiVT forecast (HiVT-only).

    HiVT futures are defined relative to observation time T_obs
    (history end / positions[hist-1]), NOT relative to inference completion:
      future_times[k] = T_obs + (k + 1) * dt

    Default consumer clock is HiVT ready time:
      T_ready = T_obs + L_s4d + L_hivt

    Steps with future_times < T_ready are already past for that consumer and
    are dropped.

    Limitation
    ----------
    This only aligns HiVT outputs. For Bench2Drive + Sparse4D + HiVT on one
    clock, use integration.risk_fusion.build_risk_packet.
    """
    observation_time: float
    sparse4d_latency_s: float
    hivt_latency_s: float
    total_latency_s: float
    result_ready_time: float
    dt: float
    skipped_future_steps: int
    remaining_future_steps: int
    future_abs_times: list  # absolute times for remaining steps
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sync_forecast(
    observation_time: float,
    world_xy: np.ndarray,
    mode_probabilities: np.ndarray,
    sparse4d_latency_s: float,
    hivt_latency_s: float,
    dt: float = 0.1,
    result_ready_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Drop already-elapsed forecast steps and attach absolute timestamps.

    world_xy: [F, N, H, 2] world-frame predicted centers
    mode_probabilities: [N, F]
    """
    if world_xy.ndim != 4 or world_xy.shape[-1] != 2:
        raise ValueError(f'world_xy expected [F,N,H,2], got {world_xy.shape}')
    h = world_xy.shape[2]
    if h != AV2_FUTURE_STEPS:
        pass

    total = float(sparse4d_latency_s) + float(hivt_latency_s)
    ready = float(observation_time + total) if result_ready_time is None else float(result_ready_time)
    skip = int(np.floor((ready - observation_time) / dt + 1e-9))
    skip = max(0, min(skip, h))
    remain = h - skip

    abs_times = [float(observation_time + (skip + k + 1) * dt) for k in range(remain)]
    synced = world_xy[:, :, skip:, :] if remain > 0 else world_xy[:, :, :0, :]

    meta = TimeSyncResult(
        observation_time=float(observation_time),
        sparse4d_latency_s=float(sparse4d_latency_s),
        hivt_latency_s=float(hivt_latency_s),
        total_latency_s=total,
        result_ready_time=ready,
        dt=float(dt),
        skipped_future_steps=skip,
        remaining_future_steps=remain,
        future_abs_times=abs_times,
        note=(
            'HiVT-only: drop past waypoints. For Bench2Drive+Sparse4D+HiVT risk fusion '
            'use integration.risk_fusion.build_risk_packet — discard-only does not '
            'align Sparse4D/B2D geometry to result_ready_time.'
        ),
    )
    return {
        'time_sync': meta.to_dict(),
        'synced_world_xy': synced,
        'mode_probabilities': mode_probabilities,
        'raw_world_xy': world_xy,
    }
