# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import ArgoverseV2Dataset
from models.hivt import HiVT


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float('nan')
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = min(len(sorted_vals) - 1, max(0, int(round((len(sorted_vals) - 1) * q))))
    return sorted_vals[idx]


def _rotate_labels_for_metrics(batch) -> None:
    """Align GT with model frame after deploy-style forward (y was ignored)."""
    y = batch.y
    rotate_mat = batch['rotate_mat'] if 'rotate_mat' in batch else None
    if y is not None and rotate_mat is not None:
        batch.y = torch.bmm(y, rotate_mat)


@torch.no_grad()
def evaluate_with_latency(model: HiVT,
                          dataloader: DataLoader,
                          device: torch.device,
                          warmup_batches: int = 10) -> Dict[str, float]:
    model.eval()
    model.to(device)

    forward_ms: List[float] = []
    e2e_ms: List[float] = []
    scenarios_per_batch: List[int] = []

    # Warmup (not timed).
    it = iter(dataloader)
    for _ in range(min(warmup_batches, len(dataloader))):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = batch.to(device)
        y = batch.y
        batch.y = None
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _ = model(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batch.y = y

    # Full pass for metrics + timing.
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='eval', dynamic_ncols=True)):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            e2e_starter = torch.cuda.Event(enable_timing=True)
            e2e_ender = torch.cuda.Event(enable_timing=True)

            e2e_starter.record()
            batch = batch.to(device)
            y = batch.y
            batch.y = None
            starter.record()
            y_hat, pi = model(batch)
            ender.record()
            e2e_ender.record()
            torch.cuda.synchronize()
            forward_ms.append(starter.elapsed_time(ender))
            e2e_ms.append(e2e_starter.elapsed_time(e2e_ender))
            batch.y = y
            _rotate_labels_for_metrics(batch)
        else:
            import time
            t0 = time.perf_counter()
            batch = batch.to(device)
            y = batch.y
            batch.y = None
            t1 = time.perf_counter()
            y_hat, pi = model(batch)
            t2 = time.perf_counter()
            forward_ms.append((t2 - t1) * 1000.0)
            e2e_ms.append((t2 - t0) * 1000.0)
            batch.y = y
            _rotate_labels_for_metrics(batch)

        scenarios_per_batch.append(int(batch.num_graphs))

        # Same metric path as HiVT.validation_step (focal agent / agent_index).
        y_hat_agent = y_hat[:, batch['agent_index'], :, :2]
        y_agent = batch.y[batch['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(batch.num_graphs, device=device)]
        model.minADE.update(y_hat_best_agent, y_agent)
        model.minFDE.update(y_hat_best_agent, y_agent)
        model.minMR.update(y_hat_best_agent, y_agent)
        _ = pi

    ade = float(model.minADE.compute().detach().cpu())
    fde = float(model.minFDE.compute().detach().cpu())
    mr = float(model.minMR.compute().detach().cpu())

    total_scenarios = sum(scenarios_per_batch)
    total_forward_ms = sum(forward_ms)
    total_e2e_ms = sum(e2e_ms)
    fwd_sorted = sorted(forward_ms)
    e2e_sorted = sorted(e2e_ms)

    per_scene_fwd = []
    per_scene_e2e = []
    for ms, n in zip(forward_ms, scenarios_per_batch):
        per_scene_fwd.append(ms / max(n, 1))
    for ms, n in zip(e2e_ms, scenarios_per_batch):
        per_scene_e2e.append(ms / max(n, 1))
    per_scene_fwd_sorted = sorted(per_scene_fwd)
    per_scene_e2e_sorted = sorted(per_scene_e2e)

    return {
        'val_minADE': ade,
        'val_minFDE': fde,
        'val_minMR': mr,
        'num_batches': float(len(forward_ms)),
        'num_scenarios': float(total_scenarios),
        'warmup_batches': float(warmup_batches),
        'forward_ms_mean': total_forward_ms / max(len(forward_ms), 1),
        'forward_ms_p50': _percentile(fwd_sorted, 0.50),
        'forward_ms_p95': _percentile(fwd_sorted, 0.95),
        'e2e_ms_mean': total_e2e_ms / max(len(e2e_ms), 1),
        'e2e_ms_p50': _percentile(e2e_sorted, 0.50),
        'e2e_ms_p95': _percentile(e2e_sorted, 0.95),
        'forward_ms_per_scenario_mean': sum(per_scene_fwd) / max(len(per_scene_fwd), 1),
        'forward_ms_per_scenario_p50': _percentile(per_scene_fwd_sorted, 0.50),
        'e2e_ms_per_scenario_mean': sum(per_scene_e2e) / max(len(per_scene_e2e), 1),
        'e2e_ms_per_scenario_p50': _percentile(per_scene_e2e_sorted, 0.50),
        'throughput_scenarios_per_s_forward': total_scenarios / max(total_forward_ms / 1000.0, 1e-9),
        'throughput_scenarios_per_s_e2e': total_scenarios / max(total_e2e_ms / 1000.0, 1e-9),
    }


def main() -> None:
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--warmup_batches', type=int, default=10)
    parser.add_argument('--max_scenarios', type=int, default=None,
                        help='Optional cap for latency profiling')
    parser.add_argument('--budget_ms', type=float, default=100.0,
                        help='Realtime budget per scene (AV2 10Hz -> 100ms)')
    parser.add_argument('--fast_infer', action='store_true', default=True,
                        help='Enable parallel AAEncoder path for lower latency (default: on)')
    parser.add_argument('--no_fast_infer', action='store_true',
                        help='Disable fast inference (use training-time parallel=False path)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if args.gpus and torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path)
    use_fast = args.fast_infer and not args.no_fast_infer
    if use_fast:
        model.enable_fast_inference()
    local_radius = model.hparams.local_radius
    val_dataset = ArgoverseV2Dataset(root=args.root, split='val', local_radius=local_radius,
                                     max_scenarios=args.max_scenarios)

    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == 'cuda',
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )

    results = evaluate_with_latency(
        model=model,
        dataloader=dataloader,
        device=device,
        warmup_batches=args.warmup_batches,
    )

    # Single-scene realtime view: when batch_size==1, batch latency == scene latency.
    scene_fwd_p50 = results['forward_ms_per_scenario_p50']
    scene_fwd_p95 = results['forward_ms_p95'] / max(args.batch_size, 1) if args.batch_size > 1 else results['forward_ms_p95']
    scene_e2e_p50 = results['e2e_ms_per_scenario_p50']
    scene_e2e_p95 = results['e2e_ms_p95'] / max(args.batch_size, 1) if args.batch_size > 1 else results['e2e_ms_p95']
    if args.batch_size == 1:
        scene_fwd_p50 = results['forward_ms_p50']
        scene_fwd_p95 = results['forward_ms_p95']
        scene_e2e_p50 = results['e2e_ms_p50']
        scene_e2e_p95 = results['e2e_ms_p95']

    print('\n========== FINAL EVAL ==========')
    print(f'ckpt           : {args.ckpt_path}')
    print(f'device         : {device}')
    if device.type == 'cuda':
        print(f'gpu            : {torch.cuda.get_device_name(0)}')
    print(f'dataset/split  : {args.dataset}/val')
    print(f'batch_size     : {args.batch_size}')
    print(f'fast_infer     : {use_fast} (parallel={model.parallel})')
    print('latency_mode   : deploy (forward without GT label rotation)')
    print(f'num_workers    : {args.num_workers}'
          + ('  [note: workers can inflate CUDA-event latency on WSL/shared FS; use 0 for model timing]'
             if args.num_workers > 0 else ''))
    print(f'scenarios      : {int(results["num_scenarios"])}')
    print(f'batches        : {int(results["num_batches"])} (warmup={int(results["warmup_batches"])})')
    print('--- accuracy ---')
    print(f'val_minADE     : {results["val_minADE"]:.4f}')
    print(f'val_minFDE     : {results["val_minFDE"]:.4f}')
    print(f'val_minMR      : {results["val_minMR"]:.4f}')
    print('--- latency (batch) ---')
    print(f'forward_ms     : mean={results["forward_ms_mean"]:.2f}  p50={results["forward_ms_p50"]:.2f}  p95={results["forward_ms_p95"]:.2f}')
    print(f'e2e_ms (H2D+fwd): mean={results["e2e_ms_mean"]:.2f}  p50={results["e2e_ms_p50"]:.2f}  p95={results["e2e_ms_p95"]:.2f}')
    print('--- latency (per scenario) ---')
    print(f'forward_ms/scen: mean={results["forward_ms_per_scenario_mean"]:.2f}  p50={results["forward_ms_per_scenario_p50"]:.2f}')
    print(f'e2e_ms/scen    : mean={results["e2e_ms_per_scenario_mean"]:.2f}  p50={results["e2e_ms_per_scenario_p50"]:.2f}')
    print('--- throughput ---')
    print(f'scenarios/s fwd: {results["throughput_scenarios_per_s_forward"]:.2f}')
    print(f'scenarios/s e2e: {results["throughput_scenarios_per_s_e2e"]:.2f}')
    print('--- realtime check ---')
    print(f'mode           : {"single-scene (batch_size=1)" if args.batch_size == 1 else f"batched (batch_size={args.batch_size}, amortized)"}')
    print(f'budget_ms      : {args.budget_ms:.1f}  ({1000.0 / args.budget_ms:.1f} Hz)')
    print(f'scene_fwd p50/p95 : {scene_fwd_p50:.2f} / {scene_fwd_p95:.2f} ms')
    print(f'scene_e2e p50/p95 : {scene_e2e_p50:.2f} / {scene_e2e_p95:.2f} ms')
    print(f'fwd within budget : p50={"YES" if scene_fwd_p50 <= args.budget_ms else "NO"}, p95={"YES" if scene_fwd_p95 <= args.budget_ms else "NO"}')
    print(f'e2e within budget : p50={"YES" if scene_e2e_p50 <= args.budget_ms else "NO"}, p95={"YES" if scene_e2e_p95 <= args.budget_ms else "NO"}')
    print('================================')


if __name__ == '__main__':
    main()
