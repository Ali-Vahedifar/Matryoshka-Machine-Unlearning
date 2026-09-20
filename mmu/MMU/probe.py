#!/usr/bin/env python3
import os
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY.parent))
sys.path.insert(0, str(REPOSITORY))

from Machine_Unlearning_baselines import FineTune, SCRUB
from MMU import MSCRUB
from MMU import (
    capture_features, default_granularities, linear_head,
)
import scripts.benchmark_unlearning as bench


@torch.no_grad()
def features_and_labels(model, loader, device):
    head = linear_head(model)
    model.eval()
    features, labels = [], []
    for inputs, targets in loader:
        features.append(capture_features(model, head, inputs.to(device)).cpu())
        labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)


def fit_probe(train_x, train_y, num_classes, steps=300, lr=1e-2, seed=0, device='cpu'):
    torch.manual_seed(seed)
    train_x, train_y = train_x.to(device), train_y.to(device)
    probe = torch.nn.Linear(train_x.shape[1], num_classes).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        F.cross_entropy(probe(train_x), train_y).backward()
        optimiser.step()
    return probe


@torch.no_grad()
def accuracy(probe, x, y, device='cpu'):
    predictions = probe(x.to(device)).argmax(dim=1).cpu()
    return 100.0 * float((predictions == y).float().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--reference_dir', default=str(REPOSITORY.parent / 'results/prefix_depth_runs/results_mu_cifar10_3seed'))
    parser.add_argument('--mmu_dir', default=str(REPOSITORY.parent / 'results/prefix_depth_runs/results_mmu_mrl_cifar10_3seed'))
    parser.add_argument('--mmu_e_dir', default=str(REPOSITORY.parent / 'results/prefix_depth_runs/results_mmu_cifar10_3seed'))
    options = parser.parse_args()

    def results(directory):
        return json.loads((Path(directory) / f'seed{options.seed}'
                           / 'final.json').read_text())['results']

    reference = results(options.reference_dir)
    mrl = results(options.mmu_dir)
    tied = results(options.mmu_e_dir)

    sys.argv = [
        'benchmark', '--dataset', 'cifar10', '--forget_class', '0', '--epochs', '30',
        '--batch_size', '64', '--source_lr', '0.001', '--retain_ratio', '0.1',
        '--seed', str(options.seed), '--device', options.device,
        '--data_dir', os.environ.get('MMU_DATA', 'data'), '--phase', 'prepare', '--methods', 'mmu',
        '--output_dir', str(Path(os.environ.get('MMU_WORK', 'work')) / 'prefix_depth' / f'seed{options.seed}'),
    ]
    benchmark = bench.Benchmark(bench.parse_args())
    device = options.device
    forget, retain = benchmark.loaders['forget_train'], benchmark.loaders['retain_small']
    widths = default_granularities(linear_head(benchmark.factory()).in_features)

    def unlearned(builder):
        bench.set_seed(options.seed)
        model = benchmark._fresh_source()
        if builder is not None:
            builder(model).unlearn(forget, retain)
        return model

    oracle = benchmark.factory().to(device)
    oracle.load_state_dict(benchmark.reference_state)

    models = {
        'original': unlearned(None),
        'retrain (oracle)': oracle,
        'finetune': unlearned(lambda m: FineTune(
            model=m, device=device, **reference['finetune']['config'])),
        'scrub': unlearned(lambda m: SCRUB(
            model=m, device=device, **reference['scrub']['config'])),
        'mmu (MRL)': unlearned(lambda m: MSCRUB(
            model=m, device=device, diagnose=False, **mrl['mmu']['config'])),
        'mmu (MRL-E)': unlearned(lambda m: MSCRUB(
            model=m, device=device, diagnose=False, head_mode='tied',
            weighting='normalised',
            **{k: v for k, v in tied['mmu']['config'].items()})),
    }

    rows = {}
    started = time.perf_counter()
    for name, model in models.items():
        train_x, train_y = features_and_labels(model, benchmark.loaders['forget_train'], device)
        retain_x, retain_y = features_and_labels(model, benchmark.loaders['retain_small'], device)
        train_x = torch.cat([train_x, retain_x])
        train_y = torch.cat([train_y, retain_y])
        test_f = features_and_labels(model, benchmark.loaders['forget_test'], device)
        test_r = features_and_labels(model, benchmark.loaders['retain_test'], device)
        rows[name] = {'D_f': {}, 'D_r': {}, 'deployed_D_f': {}}
        for m in widths:
            probe = fit_probe(train_x[:, :m], train_y, benchmark.num_classes,
                              steps=options.steps, seed=options.seed, device=device)
            rows[name]['D_f'][int(m)] = accuracy(probe, test_f[0][:, :m], test_f[1], device)
            rows[name]['D_r'][int(m)] = accuracy(probe, test_r[0][:, :m], test_r[1], device)
        print(f'  probed {name} ({time.perf_counter() - started:.0f}s)', flush=True)

    print(f'\nFrozen-backbone linear probe, CIFAR-10 seed {options.seed}, '
          f'{options.steps} Adam steps per probe')
    print(f'(SCRUB {reference["scrub"]["config"]}; MMU-MRL {mrl["mmu"]["config"]})')
    for split in ('D_f', 'D_r'):
        print(f'\nrecovered {split} accuracy')
        print(f'{"method":<18}' + ''.join(f'{f"m={m}":>10}' for m in widths))
        for name, values in rows.items():
            print(f'{name:<18}' + ''.join(f'{values[split][int(m)]:>10.2f}' for m in widths))
    floor = rows['retrain (oracle)']['D_f']
    print('\nexcess over the oracle floor (recovered D_f minus oracle D_f)')
    print(f'{"method":<18}' + ''.join(f'{f"m={m}":>10}' for m in widths))
    for name, values in rows.items():
        if name == 'retrain (oracle)':
            continue
        print(f'{name:<18}' + ''.join(
            f'{values["D_f"][int(m)] - floor[int(m)]:>10.2f}' for m in widths))

    (REPOSITORY.parent / 'results/mmu_prefix_leakage' / (f'mmu_probe_seed{options.seed}.json')).write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
