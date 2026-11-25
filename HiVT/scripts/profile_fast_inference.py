#!/usr/bin/env python3
"""Profile current fast-inference (parallel) LocalEncoder path."""
from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Dict, List

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import ArgoverseV2Dataset
from models.hivt import HiVT
from utils import DistanceDropEdge


def cuda_ms(fn) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


@torch.no_grad()
def attach_rotate(model: HiVT, data, device):
    data = data.to(device)
    if model.rotate:
        rotate_mat = torch.empty(data.num_nodes, 2, 2, device=device)
        sin_vals = torch.sin(data['rotate_angles'])
        cos_vals = torch.cos(data['rotate_angles'])
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals
        rotate_mat[:, 1, 0] = sin_vals
        rotate_mat[:, 1, 1] = cos_vals
        if data.y is not None:
            data.y = torch.bmm(data.y, rotate_mat)
        data['rotate_mat'] = rotate_mat
    else:
        data['rotate_mat'] = None
    return data


@torch.no_grad()
def profile_one(model: HiVT, data, device) -> Dict[str, float]:
    data = attach_rotate(model, data, device)
    local = model.local_encoder
    hist = model.historical_steps
    drop_edge = DistanceDropEdge(model.hparams.local_radius)
    stats = {
        'num_agents': float(data.num_nodes),
        'num_lanes': float(data['lane_vectors'].size(0)),
    }

    def build_aa():
        return local._build_parallel_aa_graph(data)
    stats['ms_build_aa_graph'] = cuda_ms(build_aa)
    batch_x, batch_ei, batch_ea = local._build_parallel_aa_graph(data)
    stats['num_aa_edges'] = float(batch_ei.size(1))

    def aa():
        return local.aa_encoder(
            x=batch_x, t=None, edge_index=batch_ei, edge_attr=batch_ea,
            bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
    stats['ms_aa_encoder'] = cuda_ms(aa)
    out = local.aa_encoder(
        x=batch_x, t=None, edge_index=batch_ei, edge_attr=batch_ea,
        bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
    out = out.reshape(hist, out.shape[0] // hist, -1)

    def temporal():
        return local.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, :hist])
    stats['ms_temporal'] = cuda_ms(temporal)
    out = local.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, :hist])
    if local.actor_type_embed is not None:
        out = out + local.actor_type_embed(data['actor_type'])

    edge_index, edge_attr = drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
    stats['num_lane_actor_edges'] = float(edge_index.size(1))

    def al():
        return local.al_encoder(
            x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
            is_intersections=data['is_intersections'], lane_types=data['lane_types'],
            left_mark_types=data['left_mark_types'], right_mark_types=data['right_mark_types'],
            rotate_mat=data['rotate_mat'])
    stats['ms_al'] = cuda_ms(al)
    out = al()

    def glob():
        return model.global_interactor(data=data, local_embed=out)
    stats['ms_global'] = cuda_ms(glob)
    global_embed = model.global_interactor(data=data, local_embed=out)

    def dec():
        return model.decoder(local_embed=out, global_embed=global_embed)
    stats['ms_decoder'] = cuda_ms(dec)

    parts = ['ms_build_aa_graph', 'ms_aa_encoder', 'ms_temporal', 'ms_al', 'ms_global', 'ms_decoder']
    stats['ms_sum_parts'] = sum(stats[k] for k in parts)
    return stats


@torch.no_grad()
def full_ms(model, data, device) -> float:
    data = data.to(device)
    return cuda_ms(lambda: model(data))


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default=str(ROOT / 'datasets'))
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--num_scenes', type=int, default=80)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model.enable_fast_inference()
    model.eval().to(device)
    ds = ArgoverseV2Dataset(root=args.root, split='val', local_radius=50.0,
                            max_scenarios=args.num_scenes + args.warmup)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for i, batch in enumerate(loader):
        if i >= args.warmup:
            break
        _ = full_ms(model, batch, device)

    full_times, parts = [], []
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for i, batch in enumerate(tqdm(loader, total=args.warmup + args.num_scenes, dynamic_ncols=True)):
        if i < args.warmup:
            continue
        if i >= args.warmup + args.num_scenes:
            break
        full_times.append(full_ms(model, batch, device))
        parts.append(profile_one(model, ds[i], device))

    avg = mean_dict(parts)
    full_mean = sum(full_times) / len(full_times)
    print('\n========== FAST-INFER PROFILE (bs=1) ==========')
    print(f'gpu             : {torch.cuda.get_device_name(0)}')
    print(f'full forward ms : mean={full_mean:.2f}')
    print(f'avg agents/lanes: {avg["num_agents"]:.1f} / {avg["num_lanes"]:.1f}')
    print(f'avg AA edges    : {avg["num_aa_edges"]:.1f}')
    print(f'avg AL edges    : {avg["num_lane_actor_edges"]:.1f}')
    for name, key in [
        ('build_aa_graph', 'ms_build_aa_graph'),
        ('AAEncoder', 'ms_aa_encoder'),
        ('TemporalEncoder', 'ms_temporal'),
        ('ALEncoder', 'ms_al'),
        ('GlobalInteractor', 'ms_global'),
        ('Decoder', 'ms_decoder'),
        ('sum_parts', 'ms_sum_parts'),
    ]:
        ms = avg[key]
        print(f'{name:18s}: {ms:7.2f} ms  ({100.0 * ms / full_mean:5.1f}%)')
    print('================================================')


if __name__ == '__main__':
    main()
