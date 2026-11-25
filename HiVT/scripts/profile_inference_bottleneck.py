#!/usr/bin/env python3
"""Profile HiVT single-scene forward to explain realtime gap vs paper (~20ms)."""
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
from torch_geometric.utils import subgraph


def cuda_time(fn) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


@torch.no_grad()
def profile_one(model: HiVT, data, device: torch.device) -> Dict[str, float]:
    data = data.to(device)
    drop_edge = DistanceDropEdge(model.hparams.local_radius)
    local = model.local_encoder
    hist = model.historical_steps

    stats = {
        'num_agents': float(data.num_nodes),
        'num_lanes': float(data['lane_vectors'].size(0)),
        'num_lane_actor_edges_raw': float(data['lane_actor_index'].size(1)),
    }

    # --- LocalEncoder pieces (mirrors LocalEncoder.forward) ---
    # Build temporal edges
    def build_edges():
        for t in range(hist):
            data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
            data[f'edge_attr_{t}'] = (
                data['positions'][data[f'edge_index_{t}'][0], t] -
                data['positions'][data[f'edge_index_{t}'][1], t])
    stats['ms_build_temporal_edges'] = cuda_time(build_edges)

    outs = [None] * hist

    def aa_all():
        for t in range(hist):
            edge_index, edge_attr = drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
            outs[t] = local.aa_encoder(
                x=data.x[:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])
    stats['ms_aa_encoder_all_T'] = cuda_time(aa_all)
    out = torch.stack(outs)

    def temporal():
        return local.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, :hist])
    # Need output for next stages
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = local.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, :hist])
    end.record()
    torch.cuda.synchronize()
    stats['ms_temporal_encoder'] = start.elapsed_time(end)

    def actor_type():
        if local.actor_type_embed is not None:
            return out + local.actor_type_embed(data['actor_type'])
        return out
    start.record()
    out = actor_type()
    end.record()
    torch.cuda.synchronize()
    stats['ms_actor_type_embed'] = start.elapsed_time(end)

    edge_index, edge_attr = drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
    stats['num_lane_actor_edges'] = float(edge_index.size(1))

    def al():
        return local.al_encoder(
            x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
            is_intersections=data['is_intersections'], lane_types=data['lane_types'],
            left_mark_types=data['left_mark_types'], right_mark_types=data['right_mark_types'],
            rotate_mat=data['rotate_mat'])
    start.record()
    out = al()
    end.record()
    torch.cuda.synchronize()
    stats['ms_al_encoder'] = start.elapsed_time(end)

    def global_interactor():
        return model.global_interactor(data=data, local_embed=out)
    start.record()
    global_embed = global_interactor()
    end.record()
    torch.cuda.synchronize()
    stats['ms_global_interactor'] = start.elapsed_time(end)

    def decoder():
        return model.decoder(local_embed=out, global_embed=global_embed)
    start.record()
    _ = decoder()
    end.record()
    torch.cuda.synchronize()
    stats['ms_decoder'] = start.elapsed_time(end)

    # Full forward for reference
    def full():
        # Fresh rotate_mat path inside model.forward mutates data.y; use a clone batch ideally.
        # Here we only time another full call on same object after previous steps.
        pass

    # Time pristine full forward on a copy-loaded path is done outside.
    parts = [
        'ms_build_temporal_edges', 'ms_aa_encoder_all_T', 'ms_temporal_encoder',
        'ms_actor_type_embed', 'ms_al_encoder', 'ms_global_interactor', 'ms_decoder'
    ]
    stats['ms_sum_parts'] = sum(stats[k] for k in parts)
    return stats


@torch.no_grad()
def full_forward_ms(model: HiVT, data, device: torch.device) -> float:
    data = data.to(device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = model(data)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default=str(ROOT / 'datasets'))
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--num_scenes', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model.eval().to(device)
    ds = ArgoverseV2Dataset(root=args.root, split='val', local_radius=50.0,
                            max_scenarios=args.num_scenes + args.warmup)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Warmup
    for i, batch in enumerate(loader):
        if i >= args.warmup:
            break
        _ = full_forward_ms(model, batch, device)

    full_times = []
    part_stats = []
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for i, batch in enumerate(tqdm(loader, total=args.warmup + args.num_scenes, dynamic_ncols=True)):
        if i < args.warmup:
            continue
        if i >= args.warmup + args.num_scenes:
            break
        # Profile parts on a fresh batch instance
        full_times.append(full_forward_ms(model, batch, device))
        # Reload same index to avoid mutate issues from rotate/y
        batch2 = ds[i]
        # Need rotate_mat like model.forward
        batch2 = batch2.to(device)
        if model.rotate:
            rotate_mat = torch.empty(batch2.num_nodes, 2, 2, device=device)
            sin_vals = torch.sin(batch2['rotate_angles'])
            cos_vals = torch.cos(batch2['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if batch2.y is not None:
                batch2.y = torch.bmm(batch2.y, rotate_mat)
            batch2['rotate_mat'] = rotate_mat
        else:
            batch2['rotate_mat'] = None
        part_stats.append(profile_one(model, batch2, device))

    avg = mean_dict(part_stats)
    print('\n========== BOTTLENECK PROFILE (bs=1) ==========')
    print(f'gpu              : {torch.cuda.get_device_name(0)}')
    print(f'hist/future      : {model.hparams.historical_steps}/{model.hparams.future_steps}')
    print(f'embed_dim        : {model.hparams.embed_dim}')
    print(f'scenes profiled  : {len(part_stats)}')
    print(f'full forward ms  : mean={sum(full_times)/len(full_times):.2f}')
    print(f'avg agents/lanes : {avg["num_agents"]:.1f} / {avg["num_lanes"]:.1f}')
    print(f'avg lane-actor E : {avg["num_lane_actor_edges"]:.1f}')
    print('--- module ms (mean) ---')
    rows = [
        ('build_temporal_edges', avg['ms_build_temporal_edges']),
        ('AAEncoder x T', avg['ms_aa_encoder_all_T']),
        ('TemporalEncoder', avg['ms_temporal_encoder']),
        ('actor_type_embed', avg['ms_actor_type_embed']),
        ('ALEncoder (map)', avg['ms_al_encoder']),
        ('GlobalInteractor', avg['ms_global_interactor']),
        ('Decoder', avg['ms_decoder']),
        ('sum_parts', avg['ms_sum_parts']),
    ]
    for name, ms in rows:
        pct = 100.0 * ms / max(sum(full_times) / len(full_times), 1e-9)
        print(f'{name:22s}: {ms:7.2f} ms  ({pct:5.1f}% of full)')
    # Complexity scaling notes vs AV1 paper setting
    t_av2 = float(model.hparams.historical_steps)
    t_av1 = 20.0
    print('--- complexity vs paper AV1 ---')
    print(f'historical_steps     : {t_av2:.0f} vs 20  (x{t_av2/t_av1:.2f})')
    print(f'future_steps         : {model.hparams.future_steps} vs 30  '
          f'(x{float(model.hparams.future_steps)/30.0:.2f})')
    print(f'AAEncoder calls      : scales ~O(T); T ratio x{t_av2/t_av1:.2f}')
    print(f'TemporalEncoder attn : scales ~O(T^2); T^2 ratio x{(t_av2/t_av1)**2:.2f}')
    print('===============================================')


if __name__ == '__main__':
    main()
