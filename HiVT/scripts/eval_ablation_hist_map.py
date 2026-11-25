#!/usr/bin/env python3
"""Ablation: short history (1-4s) and no-map accuracy on AV2 val."""
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import sys
from typing import Dict, Optional

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import ArgoverseV2Dataset
from datasets.av2_constants import AV2_FUTURE_STEPS, AV2_HISTORICAL_STEPS
from models.hivt import HiVT
from metrics import ADE, FDE, MR


def truncate_history(data, keep_hist_steps: int) -> None:
    """Simulate a track that only exists for the last `keep_hist_steps` frames.

    At 10Hz: 10/20/30/40 steps => 1/2/3/4 seconds. Earlier history is padded out,
    then bos_mask / x / rotate_angles are recomputed like dataset preprocessing.
    """
    hist = AV2_HISTORICAL_STEPS
    if keep_hist_steps <= 0 or keep_hist_steps > hist:
        raise ValueError(f'keep_hist_steps must be in 1..{hist}, got {keep_hist_steps}')

    padding = data['padding_mask'].clone()
    start = hist - keep_hist_steps
    # Drop older history for all agents (sudden appearance / short track).
    padding[:, :start] = True

    # Match dataset rule: <2 valid hist steps => do not supervise/predict future.
    for n in range(padding.size(0)):
        valid_hist = (~padding[n, :hist]).nonzero(as_tuple=False).view(-1)
        if valid_hist.numel() < 2:
            padding[n, hist:] = True

    data['padding_mask'] = padding

    bos = torch.zeros(padding.size(0), hist, dtype=torch.bool, device=padding.device)
    bos[:, 0] = ~padding[:, 0]
    bos[:, 1:] = padding[:, : hist - 1] & ~padding[:, 1:hist]
    data['bos_mask'] = bos

    positions = data['positions']
    x = positions[:, :hist].clone()
    x[:, 1:hist] = torch.where(
        (padding[:, : hist - 1] | padding[:, 1:hist]).unsqueeze(-1),
        torch.zeros(padding.size(0), hist - 1, 2, dtype=x.dtype, device=x.device),
        x[:, 1:hist] - x[:, : hist - 1],
    )
    x[:, 0] = 0
    data['x'] = x

    # Heading from last two valid steps inside the kept window.
    rotate_angles = data['rotate_angles'].clone()
    for n in range(padding.size(0)):
        valid_hist = (~padding[n, :hist]).nonzero(as_tuple=False).view(-1)
        if valid_hist.numel() > 1:
            i0 = int(valid_hist[-2])
            i1 = int(valid_hist[-1])
            heading = positions[n, i1] - positions[n, i0]
            rotate_angles[n] = torch.atan2(heading[1], heading[0])
    data['rotate_angles'] = rotate_angles


def strip_map(data) -> None:
    """Remove all map inputs (empty lane graph)."""
    device = data['lane_vectors'].device
    data['lane_vectors'] = torch.zeros(0, 2, dtype=torch.float, device=device)
    data['is_intersections'] = torch.zeros(0, dtype=torch.uint8, device=device)
    data['lane_types'] = torch.zeros(0, dtype=torch.uint8, device=device)
    data['left_mark_types'] = torch.zeros(0, dtype=torch.uint8, device=device)
    data['right_mark_types'] = torch.zeros(0, dtype=torch.uint8, device=device)
    data['lane_actor_index'] = torch.empty(2, 0, dtype=torch.long, device=device)
    data['lane_actor_vectors'] = torch.zeros(0, 2, dtype=torch.float, device=device)


def apply_ablation_to_batch(batch: Batch,
                            keep_hist_steps: Optional[int],
                            drop_map: bool) -> Batch:
    """Apply ablation on each graph in a PyG batch (in-place on a clone)."""
    out = batch.clone()
    # Batch stores concatenated tensors; apply via per-graph split for safety.
    data_list = out.to_data_list()
    for data in data_list:
        if keep_hist_steps is not None:
            truncate_history(data, keep_hist_steps)
        if drop_map:
            strip_map(data)
    return Batch.from_data_list(data_list)


@torch.no_grad()
def eval_setting(model: HiVT,
                 dataloader: DataLoader,
                 device: torch.device,
                 keep_hist_steps: Optional[int] = None,
                 drop_map: bool = False,
                 desc: str = 'eval') -> Dict[str, float]:
    model.eval()
    ade_m, fde_m, mr_m = ADE().to(device), FDE().to(device), MR().to(device)
    n_scenes = 0
    n_skipped = 0  # focal future fully padded after ablation

    for batch in tqdm(dataloader, desc=desc, dynamic_ncols=True):
        batch = batch.to(device)
        if keep_hist_steps is not None or drop_map:
            batch = apply_ablation_to_batch(batch, keep_hist_steps, drop_map).to(device)

        y = batch.y
        batch.y = None
        y_hat, _ = model(batch)
        # rotate_mat is built inside model.forward; align GT to agent frame for metrics.
        batch.y = torch.bmm(y, batch['rotate_mat'])

        # In a PyG Batch, agent_index is already offset to global node ids.
        agent_index = batch['agent_index'].view(-1)
        for g in range(batch.num_graphs):
            a = int(agent_index[g])
            # Skip if focal has no valid future (e.g. <2 hist after truncate).
            fut_pad = batch['padding_mask'][a, AV2_HISTORICAL_STEPS:]
            if bool(fut_pad.all()):
                n_skipped += 1
                continue

            y_hat_agent = y_hat[:, a, :, :2]  # [F, H, 2]
            y_agent = batch.y[a]  # [H, 2]
            fde = torch.norm(y_hat_agent[:, -1] - y_agent[-1], p=2, dim=-1)
            best = int(fde.argmin())
            y_best = y_hat_agent[best]
            ade_m.update(y_best.unsqueeze(0), y_agent.unsqueeze(0))
            fde_m.update(y_best.unsqueeze(0), y_agent.unsqueeze(0))
            mr_m.update(y_best.unsqueeze(0), y_agent.unsqueeze(0))
            n_scenes += 1

    return {
        'val_minADE': float(ade_m.compute().detach().cpu()),
        'val_minFDE': float(fde_m.compute().detach().cpu()),
        'val_minMR': float(mr_m.compute().detach().cpu()),
        'num_scenarios': float(n_scenes),
        'num_skipped_focal': float(n_skipped),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default=str(ROOT / 'datasets'))
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--max_scenarios', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda:0' if args.gpus and torch.cuda.is_available() else 'cpu')
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    model.enable_fast_inference()
    model.eval().to(device)

    ds = ArgoverseV2Dataset(root=args.root, split='val', local_radius=50.0,
                            max_scenarios=args.max_scenarios)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == 'cuda')

    settings = [
        ('baseline_5.0s_with_map', None, False),
        ('hist_1.0s_with_map', 10, False),
        ('hist_2.0s_with_map', 20, False),
        ('hist_3.0s_with_map', 30, False),
        ('hist_4.0s_with_map', 40, False),
        ('baseline_5.0s_no_map', None, True),
    ]

    print('========== ABLATION EVAL ==========')
    print(f'ckpt           : {args.ckpt_path}')
    print(f'device         : {device}')
    if device.type == 'cuda':
        print(f'gpu            : {torch.cuda.get_device_name(0)}')
    print(f'max_scenarios  : {args.max_scenarios}')
    print(f'batch_size     : {args.batch_size}')
    print('-----------------------------------')

    rows = []
    for name, keep, drop_map in settings:
        res = eval_setting(model, loader, device, keep_hist_steps=keep, drop_map=drop_map, desc=name)
        rows.append((name, res))
        print(f'{name:28s}  ADE={res["val_minADE"]:.4f}  FDE={res["val_minFDE"]:.4f}  '
              f'MR={res["val_minMR"]:.4f}  n={int(res["num_scenarios"])}  '
              f'skip_focal={int(res["num_skipped_focal"])}')

    if rows:
        base = rows[0][1]
        print('-----------------------------------')
        print('delta vs baseline (ADE/FDE/MR):')
        for name, res in rows[1:]:
            print(f'{name:28s}  dADE={res["val_minADE"] - base["val_minADE"]:+.4f}  '
                  f'dFDE={res["val_minFDE"] - base["val_minFDE"]:+.4f}  '
                  f'dMR={res["val_minMR"] - base["val_minMR"]:+.4f}')
    print('===================================')


if __name__ == '__main__':
    main()
