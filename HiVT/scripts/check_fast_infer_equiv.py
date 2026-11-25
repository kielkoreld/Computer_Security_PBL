#!/usr/bin/env python3
"""Compare sequential AAEncoder vs parallel (vectorized) outputs/metrics."""
from pathlib import Path
import sys

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import ArgoverseV2Dataset
from models.hivt import HiVT
from utils import DistanceDropEdge


def set_parallel(model, enabled: bool):
    model.parallel = enabled
    model.local_encoder.parallel = enabled
    model.local_encoder.aa_encoder.parallel = enabled


@torch.no_grad()
def classic_parallel_local(local, data):
    """Original parallel path: per-t subgraph + Batch.from_data_list."""
    hist = local.historical_steps
    snapshots = []
    for t in range(hist):
        edge_index, _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index)
        edge_attr = data['positions'][edge_index[0], t] - data['positions'][edge_index[1], t]
        edge_index, edge_attr = local.drop_edge(edge_index, edge_attr)
        snapshots.append(Data(x=data.x[:, t], edge_index=edge_index, edge_attr=edge_attr,
                              num_nodes=data.num_nodes))
    batch = Batch.from_data_list(snapshots)
    out = local.aa_encoder(x=batch.x, t=None, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                           bos_mask=data['bos_mask'], rotate_mat=data['rotate_mat'])
    return out.reshape(hist, out.shape[0] // hist, -1)


@torch.no_grad()
def main():
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision('highest')
    except Exception:
        pass

    ckpt = ROOT / 'lightning_logs/version_1/checkpoints/epoch=55-step=349887.ckpt'
    model = HiVT.load_from_checkpoint(checkpoint_path=str(ckpt))
    model.eval().to(device)
    ds = ArgoverseV2Dataset(root=str(ROOT / 'datasets'), split='val', local_radius=50.0, max_scenarios=40)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    aa_diffs = []
    pred_diffs = []
    ade_seq, ade_fast = [], []

    for i, batch in enumerate(loader):
        if i >= 30:
            break
        raw = ds[i].to(device)
        # build rotate_mat like model.forward
        rotate_mat = torch.empty(raw.num_nodes, 2, 2, device=device)
        sin_vals = torch.sin(raw['rotate_angles'])
        cos_vals = torch.cos(raw['rotate_angles'])
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals
        rotate_mat[:, 1, 0] = sin_vals
        rotate_mat[:, 1, 1] = cos_vals
        raw['rotate_mat'] = rotate_mat
        y = raw.y
        raw.y = None

        # AA graph: classic parallel vs vectorized
        set_parallel(model, True)
        out_vec = model.local_encoder._build_parallel_aa_graph(raw)
        out_aa_vec = model.local_encoder.aa_encoder(
            x=out_vec[0], t=None, edge_index=out_vec[1], edge_attr=out_vec[2],
            bos_mask=raw['bos_mask'], rotate_mat=raw['rotate_mat'])
        out_aa_vec = out_aa_vec.reshape(model.historical_steps, -1, out_aa_vec.size(-1))
        out_aa_classic = classic_parallel_local(model.local_encoder, raw)
        aa_diffs.append((out_aa_vec - out_aa_classic).abs().max().item())

        # Full predict: sequential vs fast
        b1 = ds[i].to(device)
        b1.y = None
        set_parallel(model, False)
        y_seq, _ = model(b1)

        b2 = ds[i].to(device)
        b2.y = None
        set_parallel(model, True)
        y_fast, _ = model(b2)
        pred_diffs.append((y_seq - y_fast).abs().max().item())

        # Focal ADE-like
        y_rot = torch.bmm(y, b2['rotate_mat'])
        agent = int(b2['agent_index'])
        y_a = y_rot[agent]
        for y_hat, bucket in [(y_seq, ade_seq), (y_fast, ade_fast)]:
            y_hat_a = y_hat[:, agent, :, :2]  # [F, T, 2]
            fde = torch.norm(y_hat_a[:, -1] - y_a[-1], p=2, dim=-1)
            best = int(fde.argmin())
            y_best = y_hat_a[best]
            ade = torch.norm(y_best - y_a, p=2, dim=-1).mean().item()
            bucket.append(ade)

        print(f'scene {i}: aa_vec_vs_classic={aa_diffs[-1]:.3e} pred_seq_vs_fast={pred_diffs[-1]:.3e} '
              f'ade_seq={ade_seq[-1]:.4f} ade_fast={ade_fast[-1]:.4f}')

    print('\n=== summary ===')
    print(f'AA vectorized vs classic parallel: max={max(aa_diffs):.3e} mean={sum(aa_diffs)/len(aa_diffs):.3e}')
    print(f'pred sequential vs fast parallel:  max={max(pred_diffs):.3e} mean={sum(pred_diffs)/len(pred_diffs):.3e}')
    print(f'mean focal ADE seq/fast: {sum(ade_seq)/len(ade_seq):.4f} / {sum(ade_fast)/len(ade_fast):.4f}')
    # Vectorized should nearly match classic parallel.
    vec_ok = max(aa_diffs) < 1e-4
    print('vectorized_equiv:', 'PASS' if vec_ok else 'FAIL')


if __name__ == '__main__':
    main()
