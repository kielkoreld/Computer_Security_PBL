#!/usr/bin/env python3
"""Paper-style HiVT inference-speed benchmark.

HiVT CVPR'22 protocol (Inference Speed section / Table 5 notes):
  - GPU forward only (exclude disk I/O / CPU preprocess)
  - local_radius = 50 m
  - Realtime claim: batch_size = 1 scene, single forward for all agents
    HiVT-128 ~ 20 ms on RTX 2080 Ti (paper)
  - Table 5 also reports batch_size = 32 for cross-method comparison
    (paper frames this as ~avg agents/scene for agent-centric baselines)

This script mirrors that protocol on the available AV2 val split with our
trained checkpoint (AV1 raw data is not present in this workspace).
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import ArgoverseV2Dataset
from models.hivt import HiVT


def percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float('nan')
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return s[idx]


@torch.no_grad()
def bench(model: HiVT,
          loader: DataLoader,
          device: torch.device,
          warmup: int) -> dict:
    model.eval()
    model.to(device)

    # Warmup
    it = iter(loader)
    for _ in range(min(warmup, len(loader))):
        try:
            batch = next(it).to(device)
        except StopIteration:
            break
        torch.cuda.synchronize()
        _ = model(batch)
        torch.cuda.synchronize()

    times_ms: List[float] = []
    n_scenes = 0
    n_agents = 0
    for batch in tqdm(loader, desc=f'bs={loader.batch_size}', dynamic_ncols=True):
        batch = batch.to(device)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model(batch)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
        n_scenes += int(batch.num_graphs)
        n_agents += int(batch.num_nodes)

    total = sum(times_ms)
    return {
        'batch_size': loader.batch_size,
        'num_batches': len(times_ms),
        'num_scenes': n_scenes,
        'num_agents': n_agents,
        'avg_agents_per_scene': n_agents / max(n_scenes, 1),
        'mean_ms': total / max(len(times_ms), 1),
        'p50_ms': percentile(times_ms, 0.50),
        'p95_ms': percentile(times_ms, 0.95),
        'scenes_per_s': n_scenes / max(total / 1000.0, 1e-9),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default=str(ROOT / 'datasets'))
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--max_scenarios', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--budget_ms', type=float, default=20.0,
                        help='Paper realtime reference (~20ms for HiVT-128 bs=1)')
    parser.add_argument('--hz_budget_ms', type=float, default=100.0,
                        help='Practical AV loop budget at 10Hz')
    parser.add_argument('--fast_infer', action='store_true', default=True)
    parser.add_argument('--no_fast_infer', action='store_true')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA required for paper-style GPU timing'
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True

    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    # Match paper inference setting used in their speed table notes.
    model.hparams.local_radius = 50
    use_fast = args.fast_infer and not args.no_fast_infer
    if use_fast:
        model.enable_fast_inference()
    ds = ArgoverseV2Dataset(root=args.root, split='val', local_radius=50.0,
                            max_scenarios=args.max_scenarios)

    print('========== PAPER-STYLE INFERENCE BENCH ==========')
    print(f'gpu              : {torch.cuda.get_device_name(0)}')
    print(f'ckpt             : {args.ckpt_path}')
    print(f'hist/future/modes: {model.hparams.historical_steps}/'
          f'{model.hparams.future_steps}/{model.hparams.num_modes}')
    print(f'embed_dim/radius : {model.hparams.embed_dim}/{model.hparams.local_radius}')
    print(f'parallel/fast    : {model.parallel}/{use_fast}')
    print(f'scenarios        : {len(ds)}')
    print('protocol         : CUDA Event forward-only, exclude disk I/O')
    print('paper realtime   : bs=1 scene forward, HiVT-128 ~20ms (RTX 2080 Ti)')
    print('')

    results = {}
    for bs in (1, 32):
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
        results[bs] = bench(model, loader, device, warmup=args.warmup)

    r1 = results[1]
    r32 = results[32]

    print('--- Table-note realtime protocol (batch_size=1 scene) ---')
    print(f'avg agents/scene : {r1["avg_agents_per_scene"]:.2f}')
    print(f'forward_ms       : mean={r1["mean_ms"]:.2f}  p50={r1["p50_ms"]:.2f}  p95={r1["p95_ms"]:.2f}')
    print(f'scenes/s         : {r1["scenes_per_s"]:.2f}')
    print(f'vs paper 20ms    : mean={"YES" if r1["mean_ms"] <= args.budget_ms else "NO"} '
          f'(budget={args.budget_ms:.1f}ms)')
    print(f'vs 10Hz 100ms    : mean={"YES" if r1["mean_ms"] <= args.hz_budget_ms else "NO"} '
          f'(budget={args.hz_budget_ms:.1f}ms)')
    print('')
    print('--- Paper Table 5 style batch_size=32 (scenes batched here) ---')
    print(f'forward_ms/batch : mean={r32["mean_ms"]:.2f}  p50={r32["p50_ms"]:.2f}  p95={r32["p95_ms"]:.2f}')
    print(f'amortized ms/scen: {r32["mean_ms"] / 32:.2f}')
    print(f'scenes/s         : {r32["scenes_per_s"]:.2f}')
    print('NOTE: paper Table 5 bs=32 is framed for agent-centric multi-pass comparison;')
    print('      HiVT itself predicts all agents in one forward (bs=1 is the realtime claim).')
    print('=================================================')


if __name__ == '__main__':
    main()
