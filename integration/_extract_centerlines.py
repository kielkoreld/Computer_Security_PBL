#!/usr/bin/env python3
"""Extract AV2-like lane features from Bench2Drive Town*_HD_map.npz.

Maximizes HiVT map fields from HD map alone:
  - Center polylines → lane_vectors
  - TopologyType / junction flags → is_intersections
  - Broken/Solid/SolidSolid polylines → left/right_mark_types via
    (1) same-road spatial left/right of centerline, then
    (2) Center.Left / Center.Right neighbor fallback
  - lane_type defaults to VEHICLE (B2D HD map has no bike/bus/crosswalk layers)

No trigger polygons (not AV2 lane semantics).
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from av2.map.lane_segment import LaneMarkType, LaneType
from datasets.av2_constants import LANE_TYPE_TO_ID, MARK_TYPE_TO_ID

_UNKNOWN = MARK_TYPE_TO_ID[LaneMarkType.UNKNOWN]
_NONE = MARK_TYPE_TO_ID[LaneMarkType.NONE]
_VEHICLE = LANE_TYPE_TO_ID[LaneType.VEHICLE]
_MARK_TYPES = {'Broken', 'Solid', 'SolidSolid', 'Other', 'NONE'}


def carla_xy_to_world_rh(xy: np.ndarray) -> np.ndarray:
    out = np.asarray(xy, dtype=np.float32).copy()
    out[:, 1] *= -1.0
    return out


def b2d_mark_to_av2(mark_type: str, color: str) -> int:
    t = (mark_type or '').strip()
    c = (color or 'White').strip().capitalize()
    if t in ('NONE', 'None', ''):
        return _NONE
    if t == 'Center':
        return _UNKNOWN
    table = {
        ('Broken', 'White'): MARK_TYPE_TO_ID[LaneMarkType.DASHED_WHITE],
        ('Broken', 'Yellow'): MARK_TYPE_TO_ID[LaneMarkType.DASHED_YELLOW],
        ('Broken', 'Blue'): MARK_TYPE_TO_ID[LaneMarkType.DASHED_WHITE],
        ('Solid', 'White'): MARK_TYPE_TO_ID[LaneMarkType.SOLID_WHITE],
        ('Solid', 'Yellow'): MARK_TYPE_TO_ID[LaneMarkType.SOLID_YELLOW],
        ('Solid', 'Blue'): MARK_TYPE_TO_ID[LaneMarkType.SOLID_BLUE],
        ('SolidSolid', 'White'): MARK_TYPE_TO_ID[LaneMarkType.DOUBLE_SOLID_WHITE],
        ('SolidSolid', 'Yellow'): MARK_TYPE_TO_ID[LaneMarkType.DOUBLE_SOLID_YELLOW],
        ('SolidSolid', 'Blue'): MARK_TYPE_TO_ID[LaneMarkType.DOUBLE_SOLID_WHITE],
    }
    key = (t, c)
    if key in table:
        return table[key]
    if t == 'Broken':
        return MARK_TYPE_TO_ID[LaneMarkType.DASHED_WHITE]
    if t == 'Solid':
        return MARK_TYPE_TO_ID[LaneMarkType.SOLID_WHITE]
    if t == 'SolidSolid':
        return MARK_TYPE_TO_ID[LaneMarkType.DOUBLE_SOLID_WHITE]
    return _UNKNOWN


def _lane_mark_from_segs(segs) -> int:
    prefer = ('SolidSolid', 'Solid', 'Broken', 'Other', 'NONE')
    by_type = {}
    for seg in segs:
        t = seg.get('Type')
        if t in prefer and t not in by_type:
            by_type[t] = seg
    for t in prefer:
        if t in by_type:
            return b2d_mark_to_av2(t, by_type[t].get('Color', 'White'))
    return _UNKNOWN


def _resolve_side_mark(roads: Dict[Any, Dict], side: Optional[Tuple]) -> int:
    if side is None:
        return _NONE
    try:
        rid, lid = side[0], side[1]
    except Exception:
        return _UNKNOWN
    lanes = roads.get(rid)
    if not lanes:
        return _UNKNOWN
    segs = lanes.get(lid)
    if not segs:
        return _UNKNOWN
    return _lane_mark_from_segs(segs)


def _polyline_xy(seg) -> Optional[np.ndarray]:
    pts = seg.get('Points') or []
    if len(pts) < 2:
        return None
    try:
        xy = np.asarray([p[0][:2] for p in pts], dtype=np.float32)
    except Exception:
        return None
    if xy.ndim != 2 or xy.shape[0] < 2:
        return None
    return carla_xy_to_world_rh(xy)


def _spatial_side_marks(
    center_xy: np.ndarray,
    road_marks: List[Dict[str, Any]],
    max_dist: float = 3.5,
) -> Tuple[int, int]:
    """Assign left/right mark ids by geometry relative to centerline heading."""
    if not road_marks or center_xy.shape[0] < 2:
        return _UNKNOWN, _UNKNOWN
    mid_i = center_xy.shape[0] // 2
    # Local tangent near midpoint.
    i0 = max(0, mid_i - 1)
    i1 = min(center_xy.shape[0] - 1, mid_i + 1)
    tang = center_xy[i1] - center_xy[i0]
    nrm = float(np.linalg.norm(tang))
    if nrm < 1e-3:
        tang = center_xy[-1] - center_xy[0]
        nrm = float(np.linalg.norm(tang))
    if nrm < 1e-3:
        return _UNKNOWN, _UNKNOWN
    tang = tang / nrm
    origin = center_xy[mid_i]

    best_l = (_UNKNOWN, max_dist + 1.0)  # (mark_id, dist)
    best_r = (_UNKNOWN, max_dist + 1.0)
    for m in road_marks:
        rel = m['xy'] - origin[None, :]
        # Distance of each mark point to center mid; use nearest point.
        d = np.linalg.norm(rel, axis=-1)
        j = int(np.argmin(d))
        dist = float(d[j])
        if dist > max_dist or dist < 0.15:  # ignore on-top / far
            continue
        # 2D cross: tang × rel → +left / -right in RH world after Y flip still ok for side.
        cross = float(tang[0] * rel[j, 1] - tang[1] * rel[j, 0])
        mid = int(m['mark_id'])
        if mid in (_UNKNOWN,):
            continue
        if cross > 0.05:  # left
            if dist < best_l[1]:
                best_l = (mid, dist)
        elif cross < -0.05:  # right
            if dist < best_r[1]:
                best_r = (mid, dist)
    return best_l[0], best_r[0]


def extract(src: Path, dst: Path, mark_max_dist: float = 3.5) -> None:
    local = Path('/tmp') / src.name
    if not local.is_file() or local.stat().st_size != src.stat().st_size:
        print('copying to', local)
        shutil.copy2(src, local)
    t0 = time.time()
    arr = np.load(local, allow_pickle=True)['arr']
    print(f'loaded n_roads={len(arr)} in {time.time() - t0:.1f}s')

    roads: Dict[Any, Dict] = {road_id: lanes for road_id, lanes in arr}

    # Per-road marking polylines for spatial assignment.
    marks_by_road: Dict[Any, List[Dict[str, Any]]] = {}
    n_mark_poly = 0
    for road_id, lanes in roads.items():
        bag: List[Dict[str, Any]] = []
        for lid, segs in lanes.items():
            if str(lid) in ('Trigger_Volumes', 'TriggerVolumes'):
                continue
            for seg in segs:
                t = seg.get('Type')
                if t not in _MARK_TYPES or t == 'NONE':
                    continue
                xy = _polyline_xy(seg)
                if xy is None:
                    continue
                mid = b2d_mark_to_av2(t, seg.get('Color', 'White'))
                if mid == _UNKNOWN:
                    continue
                bag.append({'xy': xy, 'mark_id': mid, 'lane_id': lid, 'type': t})
                n_mark_poly += 1
        marks_by_road[road_id] = bag
    print(f'n_mark_polylines={n_mark_poly}')

    lanes_out = []
    mark_hist: Dict[int, int] = {}
    src_hist = {'spatial': 0, 'neighbor': 0, 'none': 0}

    for road_id, lanes in roads.items():
        road_marks = marks_by_road.get(road_id, [])
        for lid, segs in lanes.items():
            if str(lid) in ('Trigger_Volumes', 'TriggerVolumes'):
                continue
            for seg in segs:
                if seg.get('Type') != 'Center':
                    continue
                xy = _polyline_xy(seg)
                if xy is None:
                    continue
                topo = str(seg.get('TopologyType') or '')
                is_junc = 1 if 'Junction' in topo else 0
                pts = seg['Points']
                if pts and len(pts[0]) >= 3:
                    flags = [bool(p[2]) for p in pts if len(p) >= 3]
                    if flags and sum(flags) > len(flags) / 2:
                        is_junc = 1

                left_s, right_s = _spatial_side_marks(xy, road_marks, max_dist=mark_max_dist)
                left_n = _resolve_side_mark(roads, seg.get('Left'))
                right_n = _resolve_side_mark(roads, seg.get('Right'))

                # Prefer spatial; fall back to neighbor pointer; else NONE.
                if left_s != _UNKNOWN:
                    left_m, src_l = left_s, 'spatial'
                elif left_n not in (_UNKNOWN,):
                    left_m, src_l = left_n, 'neighbor'
                else:
                    left_m, src_l = _NONE, 'none'

                if right_s != _UNKNOWN:
                    right_m, src_r = right_s, 'spatial'
                elif right_n not in (_UNKNOWN,):
                    right_m, src_r = right_n, 'neighbor'
                else:
                    right_m, src_r = _NONE, 'none'

                src_hist[src_l] = src_hist.get(src_l, 0) + 1
                src_hist[src_r] = src_hist.get(src_r, 0) + 1
                mark_hist[left_m] = mark_hist.get(left_m, 0) + 1
                mark_hist[right_m] = mark_hist.get(right_m, 0) + 1

                lanes_out.append({
                    'road_id': road_id,
                    'lane_id': lid,
                    'xy': xy,
                    'is_intersection': is_junc,
                    'topology': topo,
                    'left_mark': left_m,
                    'right_mark': right_m,
                    'lane_type_id': _VEHICLE,
                })

    print(f'n_center={len(lanes_out)} elapsed={time.time() - t0:.1f}s')
    print('mark_id_hist', dict(sorted(mark_hist.items())))
    print('mark_source_hist', src_hist)
    if not lanes_out:
        raise SystemExit('no lanes found')

    sides = [c['left_mark'] for c in lanes_out] + [c['right_mark'] for c in lanes_out]
    n = len(sides)
    n_none = sum(1 for x in sides if x == _NONE)
    n_filled = n - n_none
    print(f'mark_fill_rate={n_filled}/{n} ({n_filled / n:.1%})')

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        road_id=np.asarray([c['road_id'] for c in lanes_out], dtype=object),
        lane_id=np.asarray([c['lane_id'] for c in lanes_out], dtype=object),
        is_intersection=np.asarray([c['is_intersection'] for c in lanes_out], dtype=np.uint8),
        topology=np.asarray([c['topology'] for c in lanes_out], dtype=object),
        left_mark=np.asarray([c['left_mark'] for c in lanes_out], dtype=np.uint8),
        right_mark=np.asarray([c['right_mark'] for c in lanes_out], dtype=np.uint8),
        lane_type_id=np.asarray([c['lane_type_id'] for c in lanes_out], dtype=np.uint8),
        xy=np.asarray([c['xy'] for c in lanes_out], dtype=object),
    )
    print('wrote', dst, 'size_mb', dst.stat().st_size / 1e6)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=Path, default=Path(__file__).resolve().parent / 'maps' / 'Town12_HD_map.npz')
    p.add_argument('--dst', type=Path, default=Path(__file__).resolve().parent / 'maps' / 'Town12_lanes_rh.npz')
    p.add_argument('--mark_max_dist', type=float, default=3.5)
    args = p.parse_args()
    extract(args.src, args.dst, mark_max_dist=args.mark_max_dist)


if __name__ == '__main__':
    main()
