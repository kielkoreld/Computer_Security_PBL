# Copyright (c) 2022, Zikang Zhou. All rights reserved.
# Extended for Argoverse 2 motion forecasting.
import os
from itertools import permutations
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.datasets.motion_forecasting.data_schema import Track
from av2.map.lane_segment import LaneMarkType
from av2.map.map_api import ArgoverseStaticMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from datasets.av2_constants import ACTOR_TYPE_TO_ID
from datasets.av2_constants import AV2_FUTURE_STEPS
from datasets.av2_constants import AV2_HISTORICAL_STEPS
from datasets.av2_constants import AV2_TOTAL_STEPS
from datasets.av2_constants import AV_TRACK_ID
from datasets.av2_constants import CROSSWALK_TYPE_ID
from datasets.av2_constants import DYNAMIC_OBJECT_TYPES
from datasets.av2_constants import LANE_TYPE_TO_ID
from datasets.av2_constants import MARK_TYPE_TO_ID
from utils import TemporalData
from utils import migrate_temporal_data


class ArgoverseV2Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50.0,
                 max_scenarios: Optional[int] = None) -> None:
        self._split = split
        self._local_radius = local_radius
        self._max_scenarios = max_scenarios
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'{split} is not a valid AV2 split')
        self.root = root
        split_dir = Path(root) / split
        processed_dir = Path(root) / split / 'processed'
        manifest_path = processed_dir / 'manifest.txt'
        # Prefer manifest so startup does not scan ~200k scenario dirs on slow mounts.
        if manifest_path.is_file():
            names = [line.strip() for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if max_scenarios is not None:
                names = names[:max_scenarios]
            self._scenario_dirs = [split_dir / name for name in names]
            self._raw_file_names = names
            self._processed_file_names = [f'{name}.pt' for name in names]
        else:
            scenario_dirs: List[Path] = []
            for path in split_dir.iterdir():
                if path.is_dir() and path.name != 'processed':
                    scenario_dirs.append(path)
                    if max_scenarios is not None and len(scenario_dirs) >= max_scenarios:
                        break
            if max_scenarios is None:
                scenario_dirs = sorted(scenario_dirs)
            self._scenario_dirs = scenario_dirs
            self._raw_file_names = [p.name for p in self._scenario_dirs]
            self._processed_file_names = [f'{name}.pt' for name in self._raw_file_names]
        self._processed_paths = [str(processed_dir / name) for name in self._processed_file_names]
        super(ArgoverseV2Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._split)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._split, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def _process(self) -> None:
        # Avoid files_exist() over ~200k paths on /mnt/c (can take hours).
        manifest_path = Path(self.processed_dir) / 'manifest.txt'
        if manifest_path.is_file() and len(self._processed_file_names) > 0:
            return
        super(ArgoverseV2Dataset, self)._process()

    def process(self) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        ok_names: List[str] = []
        for scenario_dir in tqdm(self._scenario_dirs, desc=f'process AV2 {self._split}'):
            scenario_files = list(scenario_dir.glob('scenario_*.parquet'))
            if not scenario_files:
                continue
            scenario_path = scenario_files[0]
            map_path = scenario_dir / f'log_map_archive_{scenario_dir.name}.json'
            if not map_path.is_file():
                continue
            kwargs = process_argoverse_v2(self._split, scenario_path, map_path, self._local_radius)
            if kwargs is None:
                continue
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, f'{scenario_dir.name}.pt'))
            ok_names.append(scenario_dir.name)
        manifest_path = Path(self.processed_dir) / 'manifest.txt'
        manifest_path.write_text('\n'.join(ok_names) + ('\n' if ok_names else ''), encoding='utf-8')
        self._raw_file_names = ok_names
        self._processed_file_names = [f'{name}.pt' for name in ok_names]
        self._processed_paths = [str(Path(self.processed_dir) / name) for name in self._processed_file_names]

    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int) -> Data:
        # map_location='cpu' avoids accidental GPU pinning when DataLoader workers load samples.
        data = torch.load(self.processed_paths[idx], map_location='cpu', weights_only=False)
        return migrate_temporal_data(data)


def process_argoverse_v2(split: str,
                         scenario_path: Path,
                         map_path: Path,
                         radius: float) -> Optional[Dict]:
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    if len(scenario.timestamps_ns) != AV2_TOTAL_STEPS:
        return None

    historical_steps = set(range(AV2_HISTORICAL_STEPS))
    dynamic_tracks: List[Track] = []
    for track in scenario.tracks:
        if track.object_type not in DYNAMIC_OBJECT_TYPES:
            continue
        hist_states = [s for s in track.object_states if s.timestep in historical_steps]
        if len(hist_states) == 0:
            continue
        dynamic_tracks.append(track)
    if len(dynamic_tracks) == 0:
        return None

    track_ids = [track.track_id for track in dynamic_tracks]
    if AV_TRACK_ID not in track_ids:
        # Fall back to focal agent as the scene origin when AV is absent.
        origin_track_id = scenario.focal_track_id
    else:
        origin_track_id = AV_TRACK_ID
    if scenario.focal_track_id not in track_ids:
        return None

    av_index = track_ids.index(origin_track_id)
    agent_index = track_ids.index(scenario.focal_track_id)
    num_nodes = len(dynamic_tracks)

    actor_types = torch.tensor(
        [ACTOR_TYPE_TO_ID.get(track.object_type, ACTOR_TYPE_TO_ID[ObjectType.UNKNOWN]) for track in dynamic_tracks],
        dtype=torch.long)

    # City-frame positions over the full horizon.
    positions_city = torch.zeros(num_nodes, AV2_TOTAL_STEPS, 2, dtype=torch.float)
    padding_mask = torch.ones(num_nodes, AV2_TOTAL_STEPS, dtype=torch.bool)
    for node_idx, track in enumerate(dynamic_tracks):
        for state in track.object_states:
            t = int(state.timestep)
            if t < 0 or t >= AV2_TOTAL_STEPS:
                continue
            positions_city[node_idx, t] = torch.tensor([state.position[0], state.position[1]], dtype=torch.float)
            padding_mask[node_idx, t] = False

    # Scene origin at the origin-track pose of the current time step.
    if padding_mask[av_index, AV2_HISTORICAL_STEPS - 1]:
        return None
    origin = positions_city[av_index, AV2_HISTORICAL_STEPS - 1].clone()
    if not padding_mask[av_index, AV2_HISTORICAL_STEPS - 2]:
        heading_vec = origin - positions_city[av_index, AV2_HISTORICAL_STEPS - 2]
    else:
        heading_vec = torch.tensor([1.0, 0.0])
    theta = torch.atan2(heading_vec[1], heading_vec[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]], dtype=torch.float)

    positions = torch.matmul(positions_city - origin, rotate_mat)
    # Actors invisible at the current step are not predicted.
    for node_idx in range(num_nodes):
        if padding_mask[node_idx, AV2_HISTORICAL_STEPS - 1]:
            padding_mask[node_idx, AV2_HISTORICAL_STEPS:] = True

    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    for node_idx in range(num_nodes):
        node_hist = [t for t in range(AV2_HISTORICAL_STEPS) if not padding_mask[node_idx, t]]
        if len(node_hist) > 1:
            heading_vector = positions[node_idx, node_hist[-1]] - positions[node_idx, node_hist[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:
            padding_mask[node_idx, AV2_HISTORICAL_STEPS:] = True

    bos_mask = torch.zeros(num_nodes, AV2_HISTORICAL_STEPS, dtype=torch.bool)
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:] = padding_mask[:, : AV2_HISTORICAL_STEPS - 1] & ~padding_mask[:, 1: AV2_HISTORICAL_STEPS]

    x = positions.clone()
    x[:, AV2_HISTORICAL_STEPS:] = torch.where(
        (padding_mask[:, AV2_HISTORICAL_STEPS - 1].unsqueeze(-1) | padding_mask[:, AV2_HISTORICAL_STEPS:]).unsqueeze(-1),
        torch.zeros(num_nodes, AV2_FUTURE_STEPS, 2),
        x[:, AV2_HISTORICAL_STEPS:] - x[:, AV2_HISTORICAL_STEPS - 1].unsqueeze(-2))
    x[:, 1: AV2_HISTORICAL_STEPS] = torch.where(
        (padding_mask[:, : AV2_HISTORICAL_STEPS - 1] | padding_mask[:, 1: AV2_HISTORICAL_STEPS]).unsqueeze(-1),
        torch.zeros(num_nodes, AV2_HISTORICAL_STEPS - 1, 2),
        x[:, 1: AV2_HISTORICAL_STEPS] - x[:, : AV2_HISTORICAL_STEPS - 1])
    x[:, 0] = torch.zeros(num_nodes, 2)

    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long)

    node_inds_cur = [i for i in range(num_nodes) if not padding_mask[i, AV2_HISTORICAL_STEPS - 1]]
    node_positions_cur_city = positions_city[node_inds_cur, AV2_HISTORICAL_STEPS - 1]
    static_map = ArgoverseStaticMap.from_json(map_path)
    (lane_vectors, is_intersections, lane_types, left_mark_types, right_mark_types, lane_actor_index,
     lane_actor_vectors) = get_av2_map_features(
        static_map, node_inds_cur, node_positions_cur_city, origin, rotate_mat, radius)

    y = None if split == 'test' else x[:, AV2_HISTORICAL_STEPS:]
    return {
        'x': x[:, : AV2_HISTORICAL_STEPS],
        'positions': positions,
        'edge_index': edge_index,
        'y': y,
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,
        'bos_mask': bos_mask,
        'rotate_angles': rotate_angles,
        'actor_type': actor_types,
        'lane_vectors': lane_vectors,
        'is_intersections': is_intersections,
        'lane_types': lane_types,
        'left_mark_types': left_mark_types,
        'right_mark_types': right_mark_types,
        'lane_actor_index': lane_actor_index,
        'lane_actor_vectors': lane_actor_vectors,
        'seq_id': scenario.scenario_id,
        'av_index': av_index,
        'agent_index': agent_index,
        'city': scenario.city_name,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def _polyline_to_vectors(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points_xy.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    positions = points_xy[:-1]
    vectors = points_xy[1:] - points_xy[:-1]
    return positions.astype(np.float32), vectors.astype(np.float32)


def _crosswalk_centerline(edge1: np.ndarray, edge2: np.ndarray) -> np.ndarray:
    n = min(len(edge1), len(edge2))
    if n < 2:
        pts = np.vstack([edge1, edge2])
        return pts[:, :2].astype(np.float32)
    return ((edge1[:n] + edge2[:n]) * 0.5)[:, :2].astype(np.float32)


def get_av2_map_features(static_map: ArgoverseStaticMap,
                         node_inds: List[int],
                         node_positions_city: torch.Tensor,
                         origin: torch.Tensor,
                         rotate_mat: torch.Tensor,
                         radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
    lane_positions_list: List[torch.Tensor] = []
    lane_vectors_list: List[torch.Tensor] = []
    is_intersections_list: List[torch.Tensor] = []
    lane_types_list: List[torch.Tensor] = []
    left_marks_list: List[torch.Tensor] = []
    right_marks_list: List[torch.Tensor] = []

    origin_np = origin.numpy()
    node_positions_np = node_positions_city.numpy()

    for lane_id, lane in static_map.vector_lane_segments.items():
        centerline = static_map.get_lane_segment_centerline(lane_id)[:, :2]
        # Keep lanes that pass near any current actor in city frame.
        if len(node_positions_np) > 0:
            dists = np.linalg.norm(centerline[None, :, :] - node_positions_np[:, None, :], axis=-1).min(axis=1)
            if float(dists.min()) > radius:
                continue
        centerline = (centerline - origin_np) @ rotate_mat.numpy()
        pos, vec = _polyline_to_vectors(centerline)
        if len(vec) == 0:
            continue
        count = len(vec)
        lane_positions_list.append(torch.from_numpy(pos))
        lane_vectors_list.append(torch.from_numpy(vec))
        is_intersections_list.append(torch.full((count,), int(lane.is_intersection), dtype=torch.uint8))
        lane_types_list.append(torch.full((count,), LANE_TYPE_TO_ID.get(lane.lane_type, 0), dtype=torch.uint8))
        left_marks_list.append(torch.full(
            (count,), MARK_TYPE_TO_ID.get(lane.left_mark_type, MARK_TYPE_TO_ID[LaneMarkType.UNKNOWN]),
            dtype=torch.uint8))
        right_marks_list.append(torch.full(
            (count,), MARK_TYPE_TO_ID.get(lane.right_mark_type, MARK_TYPE_TO_ID[LaneMarkType.UNKNOWN]),
            dtype=torch.uint8))

    for _, crossing in static_map.vector_pedestrian_crossings.items():
        edge1, edge2 = crossing.get_edges_2d()
        centerline = _crosswalk_centerline(np.asarray(edge1), np.asarray(edge2))
        if len(node_positions_np) > 0:
            dists = np.linalg.norm(centerline[None, :, :] - node_positions_np[:, None, :], axis=-1).min(axis=1)
            if float(dists.min()) > radius:
                continue
        centerline = (centerline - origin_np) @ rotate_mat.numpy()
        pos, vec = _polyline_to_vectors(centerline)
        if len(vec) == 0:
            continue
        count = len(vec)
        lane_positions_list.append(torch.from_numpy(pos))
        lane_vectors_list.append(torch.from_numpy(vec))
        is_intersections_list.append(torch.zeros(count, dtype=torch.uint8))
        lane_types_list.append(torch.full((count,), CROSSWALK_TYPE_ID, dtype=torch.uint8))
        unknown_mark = MARK_TYPE_TO_ID[LaneMarkType.NONE]
        left_marks_list.append(torch.full((count,), unknown_mark, dtype=torch.uint8))
        right_marks_list.append(torch.full((count,), unknown_mark, dtype=torch.uint8))

    if len(lane_vectors_list) == 0:
        empty_l = torch.zeros(0, 2, dtype=torch.float)
        empty_i = torch.zeros(0, dtype=torch.uint8)
        empty_edge = torch.empty(2, 0, dtype=torch.long)
        return empty_l, empty_i, empty_i, empty_i, empty_i, empty_edge, empty_l

    lane_positions = torch.cat(lane_positions_list, dim=0)
    lane_vectors = torch.cat(lane_vectors_list, dim=0)
    is_intersections = torch.cat(is_intersections_list, dim=0)
    lane_types = torch.cat(lane_types_list, dim=0)
    left_mark_types = torch.cat(left_marks_list, dim=0)
    right_mark_types = torch.cat(right_marks_list, dim=0)

    node_positions = torch.matmul(node_positions_city - origin, rotate_mat).float()
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = (
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1))
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return (lane_vectors, is_intersections, lane_types, left_mark_types, right_mark_types, lane_actor_index,
            lane_actor_vectors)
