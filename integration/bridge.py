"""Sparse4D detection JSONL (+10Hz poses) → AV2 HiVT window tensors."""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from integration.paths import ensure_sparse4d_paths

ensure_sparse4d_paths()

from datasets.av2_constants import AV2_HISTORICAL_STEPS  # noqa: E402
from integration.av2_adapter import DYNAMIC_CLASSES, history_tensors_av2  # noqa: E402
from integration.b2d_map import attach_b2d_map, resolve_centerlines  # noqa: E402
from tracking import CommonTracker  # noqa: E402


def collect_track_on_grid(rs: List[dict], grid: np.ndarray, ep: str, tid: str) -> List[dict]:
    """Place native 10Hz measured detections onto the HiVT 0.1s grid.

    No temporal interpolation: missing ticks stay absent and are padded by HiVT.
    """
    rs = sorted(rs, key=lambda x: x['timestamp'])
    by_t = {round(float(r['timestamp']), 6): r for r in rs}
    cls = rs[-1].get('class', 'car')
    dense = []
    for t in grid:
        key = float(t)
        r = by_t.get(key)
        if r is None:
            continue
        dense.append({
            'episode_id': ep,
            'timestamp': key,
            'track_id': tid,
            'center_xyz': list(r['center_xyz']),
            'class': r.get('class', cls),
            'coordinate_frame': 'world_rh',
            'valid_mask': True,
            'measured': True,
        })
    return dense


def _load_map(episode_id: str, map_path: Optional[str], attach_map: bool, diag: dict):
    centerlines = None
    map_status = 'disabled'
    if attach_map:
        try:
            centerlines = resolve_centerlines(episode_id, map_path=map_path)
            map_status = 'loaded' if centerlines else 'missing_centerlines_cache'
        except Exception as exc:  # noqa: BLE001
            map_status = f'error:{type(exc).__name__}:{exc}'
            centerlines = None
    diag['map_status'] = map_status
    diag['map_n_centerlines'] = 0 if not centerlines else len(centerlines)
    return centerlines


def _append_av_poses(dense, poseby, grid, episode_id, av_id):
    for t in grid:
        pose = poseby[float(t)]
        dense.append({
            'episode_id': episode_id,
            'timestamp': float(t),
            'track_id': av_id,
            'center_xyz': np.asarray(pose['ego_to_world'], dtype=float)[:3, 3].tolist(),
            'class': 'car',
            'coordinate_frame': 'world_rh',
            'valid_mask': True,
            'measured': True,
        })


def iter_av2_windows(
    episode_id: str,
    pred_path: Path,
    poses_path: Path,
    max_windows: int = 3,
    hist_steps: int = AV2_HISTORICAL_STEPS,
    diagnostics: Optional[dict] = None,
    map_path: Optional[str] = None,
    map_radius: float = 50.0,
    attach_map: bool = True,
    score_threshold: float = 0.3,
    min_measured_hits: int = 5,
) -> Iterator[dict]:
    """Yield AV2 HiVT-ready numpy arrays from 10Hz detector JSONL + 10Hz poses."""
    if max_windows < 1:
        raise ValueError('max_windows must be positive')
    dt = 0.1
    hist_span = (hist_steps - 1) * dt  # 4.9s for 50 steps
    diag = diagnostics if diagnostics is not None else {}
    diag.update(
        episode=episode_id, target_windows=max_windows, windows=0,
        frames_examined=0, rejection_counts={}, adapter_errors={},
        scan_complete=False, stop_reason=None,
        history_source='sparse4d',
        score_threshold=score_threshold,
        min_measured_hits=min_measured_hits,
        detector_hz=10,
        temporal_interpolation=False,
    )
    centerlines = _load_map(episode_id, map_path, attach_map, diag)

    def reject(reason: str) -> None:
        counts = diag['rejection_counts']
        counts[reason] = counts.get(reason, 0) + 1

    rows = [json.loads(x) for x in Path(pred_path).read_text(encoding='utf-8').splitlines() if x.strip()]
    poses = json.loads(Path(poses_path).read_text(encoding='utf-8'))
    poseby = {round(float(x['timestamp']), 6): x for x in poses}
    tracker = CommonTracker(score_threshold=score_threshold)
    hist: List[dict] = []
    count = 0
    av_id = episode_id + ':AV'
    class_by_tid: Dict[str, str] = {}
    score_by_tid: Dict[str, float] = {}

    for i, row in enumerate(rows):
        diag['frames_examined'] += 1
        end = float(row['timestamp'])
        tracks = tracker.update(episode_id, end, row['objects'])
        for t in tracks:
            class_by_tid[t['track_id']] = t['class']
            score_by_tid[t['track_id']] = max(score_by_tid.get(t['track_id'], 0.0), float(t.get('score', 0.0)))
        hist.extend(tracks)
        hist = [x for x in hist if x['timestamp'] >= end - (hist_span + 0.11)]

        # Need a full AV2 history span before the first window is meaningful.
        if end + 1e-9 < hist_span:
            reject('warmup')
            continue

        grouped = collections.defaultdict(list)
        for x in hist:
            if x.get('valid_mask', True) and x.get('class') in DYNAMIC_CLASSES:
                grouped[x['track_id']].append(x)

        grid = np.round(end - hist_span + dt * np.arange(hist_steps), 6)
        dense: List[dict] = []
        measured_hits: Dict[str, int] = {}
        for tid, rs in grouped.items():
            if len(rs) < min_measured_hits:
                continue
            part = collect_track_on_grid(rs, grid, episode_id, tid)
            measured_hits[tid] = len(part)
            if measured_hits[tid] < min_measured_hits:
                continue
            dense.extend(part)

        if any(float(t) not in poseby for t in grid):
            reject('missing_av_pose')
            continue
        _append_av_poses(dense, poseby, grid, episode_id, av_id)

        occurrences = collections.Counter(x['track_id'] for x in dense)
        current_ids = {
            x['track_id'] for x in dense
            if x['valid_mask'] and abs(x['timestamp'] - end) <= 1e-6
        }
        cars = [x['track_id'] for x in tracks if x.get('class') == 'car']
        if not cars:
            reject('no_current_car')
            continue
        agents = [
            tid for tid in cars
            if occurrences[tid] >= min_measured_hits and tid in current_ids
            and measured_hits.get(tid, 0) >= min_measured_hits
        ]
        if not agents:
            reject('insufficient_current_car_history')
            continue
        # Prefer longest / highest-score track as focal agent (AV2-style single focal).
        agents.sort(key=lambda tid: (measured_hits.get(tid, 0), score_by_tid.get(tid, 0.0)), reverse=True)

        try:
            arrays = history_tensors_av2(
                dense, episode_id, end, av_id, agents[0],
                track_classes=class_by_tid,
            )
            if centerlines is not None:
                arrays = attach_b2d_map(arrays, centerlines, radius=map_radius)
        except ValueError as error:
            reject('adapter_rejected')
            msg = str(error)
            diag['adapter_errors'][msg] = diag['adapter_errors'].get(msg, 0) + 1
            continue

        if not arrays['prediction_eligible'][arrays['agent_index']]:
            reject('ineligible_agent')
            continue

        count += 1
        diag['windows'] = count
        if count == max_windows:
            diag.update(stop_reason='target_reached', scan_complete=False)
        yield arrays
        if count == max_windows:
            return

    diag.update(scan_complete=True, stop_reason='input_exhausted')


def iter_av2_windows_gt(
    episode_id: str,
    gt_common_path: Path,
    poses_path: Path,
    max_windows: int = 10,
    hist_steps: int = AV2_HISTORICAL_STEPS,
    diagnostics: Optional[dict] = None,
    map_path: Optional[str] = None,
    map_radius: float = 50.0,
    attach_map: bool = True,
    min_measured_hits: int = 5,
    frame_stride: int = 1,
) -> Iterator[dict]:
    """Oracle history: B2D gt_common tracks on the native 10Hz grid (perception upper bound)."""
    if max_windows < 1:
        raise ValueError('max_windows must be positive')
    dt = 0.1
    hist_span = (hist_steps - 1) * dt
    diag = diagnostics if diagnostics is not None else {}
    diag.update(
        episode=episode_id, target_windows=max_windows, windows=0,
        frames_examined=0, rejection_counts={}, adapter_errors={},
        scan_complete=False, stop_reason=None,
        history_source='gt_common',
        min_measured_hits=min_measured_hits,
        detector_hz=10,
        temporal_interpolation=False,
    )
    centerlines = _load_map(episode_id, map_path, attach_map, diag)

    def reject(reason: str) -> None:
        counts = diag['rejection_counts']
        counts[reason] = counts.get(reason, 0) + 1

    rows = [json.loads(x) for x in Path(gt_common_path).read_text(encoding='utf-8').splitlines() if x.strip()]
    poses = json.loads(Path(poses_path).read_text(encoding='utf-8'))
    poseby = {round(float(x['timestamp']), 6): x for x in poses}
    hist: List[dict] = []
    count = 0
    av_id = episode_id + ':AV'
    class_by_tid: Dict[str, str] = {}

    for i, row in enumerate(rows):
        if frame_stride > 1 and (i % frame_stride) != 0:
            continue
        diag['frames_examined'] += 1
        end = float(row['timestamp'])
        tracks = []
        for o in row.get('objects', []):
            if not o.get('valid_mask', True):
                continue
            cls = o.get('class')
            if cls not in DYNAMIC_CLASSES:
                continue
            tid = str(o['track_id'])
            class_by_tid[tid] = cls
            tracks.append({
                'episode_id': episode_id,
                'timestamp': end,
                'track_id': tid,
                'center_xyz': list(o['center_xyz']),
                'class': cls,
                'coordinate_frame': 'world_rh',
                'valid_mask': True,
                'score': 1.0,
            })
        hist.extend(tracks)
        hist = [x for x in hist if x['timestamp'] >= end - (hist_span + 0.11)]

        if end + 1e-9 < hist_span:
            reject('warmup')
            continue

        grouped = collections.defaultdict(list)
        for x in hist:
            grouped[x['track_id']].append(x)

        grid = np.round(end - hist_span + dt * np.arange(hist_steps), 6)
        if any(float(t) not in poseby for t in grid):
            reject('missing_av_pose')
            continue

        dense: List[dict] = []
        measured_hits: Dict[str, int] = {}
        for tid, rs in grouped.items():
            if len(rs) < min_measured_hits:
                continue
            part = collect_track_on_grid(rs, grid, episode_id, tid)
            measured_hits[tid] = len(part)
            if measured_hits[tid] < min_measured_hits:
                continue
            dense.extend(part)
        _append_av_poses(dense, poseby, grid, episode_id, av_id)

        occurrences = collections.Counter(x['track_id'] for x in dense)
        current_ids = {
            x['track_id'] for x in dense
            if x['valid_mask'] and abs(x['timestamp'] - end) <= 1e-6
        }
        cars = [t['track_id'] for t in tracks if t.get('class') == 'car']
        agents = [
            tid for tid in cars
            if occurrences[tid] >= min_measured_hits and tid in current_ids
            and measured_hits.get(tid, 0) >= min_measured_hits
        ]
        if not agents:
            reject('insufficient_current_car_history')
            continue
        agents.sort(key=lambda tid: measured_hits.get(tid, 0), reverse=True)

        try:
            arrays = history_tensors_av2(
                dense, episode_id, end, av_id, agents[0],
                track_classes=class_by_tid,
            )
            if centerlines is not None:
                arrays = attach_b2d_map(arrays, centerlines, radius=map_radius)
        except ValueError as error:
            reject('adapter_rejected')
            msg = str(error)
            diag['adapter_errors'][msg] = diag['adapter_errors'].get(msg, 0) + 1
            continue

        if not arrays['prediction_eligible'][arrays['agent_index']]:
            reject('ineligible_agent')
            continue

        count += 1
        diag['windows'] = count
        if count == max_windows:
            diag.update(stop_reason='target_reached', scan_complete=False)
        yield arrays
        if count == max_windows:
            return

    diag.update(scan_complete=True, stop_reason='input_exhausted')
