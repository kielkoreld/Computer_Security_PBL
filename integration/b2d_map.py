"""Bench2Drive Town HD-map → AV2 HiVT map tensors (online local query).

Onboard map package: `integration/maps/TownXX_lanes_rh.npz`
(built by `_extract_centerlines.py` from Bench2Drive-Map `TownXX_HD_map.npz`).

Load once per process; each HiVT window crops radius around current actors and
fills lane_vectors / marks / intersections (AV2 LocalEncoder contract).
"""
from __future__ import annotations

import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from integration.paths import ensure_import_paths

ensure_import_paths()

from av2.map.lane_segment import LaneMarkType
from datasets.av2_constants import CROSSWALK_TYPE_ID, MARK_TYPE_TO_ID

_UNKNOWN_MARK = MARK_TYPE_TO_ID[LaneMarkType.UNKNOWN]
_NONE_MARK = MARK_TYPE_TO_ID.get(LaneMarkType.NONE, _UNKNOWN_MARK)

DEFAULT_MAP_DIR = Path(__file__).resolve().parent / 'maps'


def town_from_episode(episode_id: str) -> Optional[str]:
    m = re.search(r'(Town\d+)', episode_id)
    return m.group(1) if m else None


def lanes_path_for_town(town: str, map_dir: Path = DEFAULT_MAP_DIR) -> Path:
    rich = Path(map_dir) / f'{town}_lanes_rh.npz'
    if rich.is_file():
        return rich
    # Backward-compatible fallback.
    return Path(map_dir) / f'{town}_centerlines_rh.npz'


def centerlines_path_for_town(town: str, map_dir: Path = DEFAULT_MAP_DIR) -> Path:
    return lanes_path_for_town(town, map_dir)


def _polyline_to_vectors(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points_xy.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    positions = points_xy[:-1]
    vectors = points_xy[1:] - points_xy[:-1]
    return positions.astype(np.float32), vectors.astype(np.float32)


def _decimate_polyline(xy: np.ndarray, step_m: float = 1.0) -> np.ndarray:
    """Keep endpoints; drop intermediate points closer than step_m."""
    if xy.shape[0] <= 2 or step_m <= 0:
        return xy
    keep = [0]
    acc = 0.0
    for i in range(1, xy.shape[0] - 1):
        acc += float(np.linalg.norm(xy[i] - xy[i - 1]))
        if acc >= step_m:
            keep.append(i)
            acc = 0.0
    keep.append(xy.shape[0] - 1)
    return xy[np.asarray(keep, dtype=np.int64)]


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


@lru_cache(maxsize=4)
def load_centerlines(path: str) -> Tuple[Dict, ...]:
    """Load town map package once; copy to /tmp for faster WSL+NTFS reads."""
    src = Path(path)
    local = Path('/tmp') / src.name
    if (not local.is_file()) or local.stat().st_size != src.stat().st_size:
        shutil.copy2(src, local)
    z = np.load(str(local), allow_pickle=True)
    keys = set(z.files)
    xys = z['xy']
    is_int = z['is_intersection']
    left = z['left_mark'] if 'left_mark' in keys else None
    right = z['right_mark'] if 'right_mark' in keys else None
    ltype = z['lane_type_id'] if 'lane_type_id' in keys else None
    out: List[Dict] = []
    lane_ids = z['lane_id'] if 'lane_id' in keys else None
    for i in range(len(xys)):
        # Drop StopSign/TrafficLight trigger polygons — not AV2 lane semantics.
        if lane_ids is not None and str(lane_ids[i]).startswith('trigger:'):
            continue
        xy = np.asarray(xys[i], dtype=np.float32)
        if xy.ndim != 2 or xy.shape[0] < 2:
            continue
        lm = int(left[i]) if left is not None else _UNKNOWN_MARK
        rm = int(right[i]) if right is not None else _UNKNOWN_MARK
        # UNKNOWN marks → NONE so AL encoder uses a trained "no mark" slot, not garbage.
        if lm == _UNKNOWN_MARK:
            lm = _NONE_MARK
        if rm == _UNKNOWN_MARK:
            rm = _NONE_MARK
        out.append({
            'xy': xy,
            'xmin': float(xy[:, 0].min()),
            'xmax': float(xy[:, 0].max()),
            'ymin': float(xy[:, 1].min()),
            'ymax': float(xy[:, 1].max()),
            'is_intersection': int(is_int[i]),
            'left_mark': lm,
            'right_mark': rm,
            'lane_type_id': int(ltype[i]) if ltype is not None else 0,
        })
    return tuple(out)


def build_av2_map_features(
    centerlines: Sequence[Dict],
    node_positions_world: np.ndarray,
    origin: np.ndarray,
    theta: float,
    radius: float = 50.0,
    lane_type_id: int = 0,
    decimate_step_m: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Build AV2 map fields in AV-local frame (same contract as get_av2_map_features)."""
    origin = np.asarray(origin, dtype=np.float64).reshape(2)
    nodes = np.asarray(node_positions_world, dtype=np.float64).reshape(-1, 2)
    if nodes.shape[0] == 0:
        return empty_av2_map()

    c, s = np.cos(theta), np.sin(theta)
    rotate = np.array([[c, -s], [s, c]], dtype=np.float64)

    nx0, ny0 = float(nodes[:, 0].min()) - radius, float(nodes[:, 1].min()) - radius
    nx1, ny1 = float(nodes[:, 0].max()) + radius, float(nodes[:, 1].max()) + radius

    lane_pos_list: List[np.ndarray] = []
    lane_vec_list: List[np.ndarray] = []
    is_int_list: List[np.ndarray] = []
    lane_type_list: List[np.ndarray] = []
    left_list: List[np.ndarray] = []
    right_list: List[np.ndarray] = []

    for lane in centerlines:
        if lane['xmax'] < nx0 or lane['xmin'] > nx1 or lane['ymax'] < ny0 or lane['ymin'] > ny1:
            continue
        xy = np.asarray(lane['xy'], dtype=np.float64)
        dists = np.linalg.norm(xy[None, :, :] - nodes[:, None, :], axis=-1).min(axis=1)
        if float(dists.min()) > radius:
            continue
        xy = _decimate_polyline(xy.astype(np.float32), step_m=decimate_step_m).astype(np.float64)
        local = (xy - origin) @ rotate
        pos, vec = _polyline_to_vectors(local.astype(np.float32))
        if len(vec) == 0:
            continue
        n = len(vec)
        lane_type = int(lane.get('lane_type_id', lane_type_id))
        if lane_type == CROSSWALK_TYPE_ID:
            is_j, lm, rm = 0, _NONE_MARK, _NONE_MARK
        else:
            is_j = int(lane.get('is_intersection', 0))
            lm = int(lane.get('left_mark', _UNKNOWN_MARK))
            rm = int(lane.get('right_mark', _UNKNOWN_MARK))
        lane_pos_list.append(pos)
        lane_vec_list.append(vec)
        is_int_list.append(np.full(n, is_j, dtype=np.uint8))
        lane_type_list.append(np.full(n, lane_type, dtype=np.uint8))
        left_list.append(np.full(n, lm, dtype=np.uint8))
        right_list.append(np.full(n, rm, dtype=np.uint8))

    if not lane_vec_list:
        return empty_av2_map()

    lane_positions = np.concatenate(lane_pos_list, axis=0)
    lane_vectors = np.concatenate(lane_vec_list, axis=0)
    is_intersections = np.concatenate(is_int_list, axis=0)
    lane_types = np.concatenate(lane_type_list, axis=0)
    left_mark_types = np.concatenate(left_list, axis=0)
    right_mark_types = np.concatenate(right_list, axis=0)

    n_lanes = lane_vectors.shape[0]
    n_nodes = nodes.shape[0]
    nodes_local = (nodes - origin) @ rotate
    lane_idx = np.repeat(np.arange(n_lanes, dtype=np.int64), n_nodes)
    actor_idx = np.tile(np.arange(n_nodes, dtype=np.int64), n_lanes)
    lane_actor_vectors = (
        lane_positions.repeat(n_nodes, axis=0) - np.tile(nodes_local, (n_lanes, 1))
    ).astype(np.float32)
    mask = np.linalg.norm(lane_actor_vectors, axis=-1) < radius
    lane_actor_index = np.stack([lane_idx[mask], actor_idx[mask]], axis=0)
    lane_actor_vectors = lane_actor_vectors[mask]

    return {
        'lane_vectors': lane_vectors.astype(np.float32),
        'is_intersections': is_intersections,
        'lane_types': lane_types,
        'left_mark_types': left_mark_types,
        'right_mark_types': right_mark_types,
        'lane_actor_index': lane_actor_index,
        'lane_actor_vectors': lane_actor_vectors.astype(np.float32),
    }


def resolve_centerlines(
    episode_id: str,
    map_path: Optional[str] = None,
    map_dir: Path = DEFAULT_MAP_DIR,
) -> Optional[Tuple[Dict, ...]]:
    if map_path:
        p = Path(map_path)
        if not p.is_file():
            raise FileNotFoundError(p)
        return load_centerlines(str(p))
    town = town_from_episode(episode_id)
    if not town:
        return None
    p = lanes_path_for_town(town, map_dir)
    if not p.is_file():
        return None
    return load_centerlines(str(p))


def attach_b2d_map(
    arrays: Dict,
    centerlines: Optional[Sequence[Dict]],
    radius: float = 50.0,
) -> Dict:
    """Replace empty map fields on an AV2 adapter window with B2D Town lanes."""
    if not centerlines:
        return arrays
    from datasets.av2_constants import AV2_HISTORICAL_STEPS

    hist = AV2_HISTORICAL_STEPS
    mask = arrays['padding_mask']
    origin = np.asarray(arrays['origin']).reshape(2)
    theta = float(arrays['theta'])
    c, s = np.cos(theta), np.sin(theta)
    scene = np.array([[c, -s], [s, c]], dtype=np.float64)
    local = arrays['positions'][:, hist - 1, :2].astype(np.float64)
    valid = ~mask[:, hist - 1]
    world = local @ scene.T + origin
    node_world = world[valid]
    feats = build_av2_map_features(
        centerlines,
        node_positions_world=node_world,
        origin=origin,
        theta=theta,
        radius=radius,
        decimate_step_m=1.0,
    )
    out = dict(arrays)
    out.update(feats)
    out['map_source'] = 'b2d_town_lanes_rich'
    out['map_n_lane_vectors'] = int(feats['lane_vectors'].shape[0])
    out['map_n_lane_actor_edges'] = int(feats['lane_actor_index'].shape[1])
    out['map_n_intersection_vecs'] = int(feats['is_intersections'].sum())
    return out
