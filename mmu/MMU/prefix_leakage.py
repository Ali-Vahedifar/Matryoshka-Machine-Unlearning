#!/usr/bin/env python3
import os
import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY.parent))
sys.path.insert(0, str(REPOSITORY))

from Machine_Unlearning_baselines import FineTune, SCRUB
from MMU import MSCRUB
from MMU import default_granularities, linear_head, prefix_accuracy
import scripts.benchmark_unlearning as bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reference_dir', default=str(REPOSITORY.parent / 'results/prefix_depth_runs/results_mu_cifar10_3seed'))
    parser.add_argument('--mmu_dir', default=str(REPOSITORY.parent / 'results/prefix_depth_runs/results_mmu_cifar10_3seed'))
    options = parser.parse_args()

    reference = json.loads(
        (Path(options.reference_dir) / f'seed{options.seed}/final.json').read_text())['results']
    mmu_results = json.loads(
        (Path(options.mmu_dir) / f'seed{options.seed}/final.json').read_text())['results']

    sys.argv = [
        'benchmark', '--dataset', 'cifar10', '--forget_class', '0', '--epochs', '30',
        '--batch_size', '64', '--source_lr', '0.001', '--retain_ratio', '0.1',
        '--seed', str(options.seed), '--device', options.device,
        '--data_dir', os.environ.get('MMU_DATA', 'data'), '--phase', 'prepare', '--methods', 'mmu',
        '--output_dir', str(Path(os.environ.get('MMU_WORK', 'work')) / 'prefix_depth' / f'seed{options.seed}'),
    ]
    benchmark = bench.Benchmark(bench.parse_args())
    forget, retain = benchmark.loaders['forget_train'], benchmark.loaders['retain_small']
    widths = default_granularities(linear_head(benchmark.factory()).in_features)

    def report(name, model):
        bench.set_seed(options.seed)
        rows[name] = {
            'D_f (train)': prefix_accuracy(model, forget, widths, options.device),
            'D_f (test)': prefix_accuracy(model, benchmark.loaders['forget_test'],
                                          widths, options.device),
            'D_r (test)': prefix_accuracy(model, benchmark.loaders['retain_test'],
                                          widths, options.device),
        }

    rows = {}
    report('original', benchmark._fresh_source())

    oracle = benchmark.factory().to(options.device)
    oracle.load_state_dict(benchmark.reference_state)
    report('retrain (oracle)', oracle)

    bench.set_seed(options.seed)
    model = benchmark._fresh_source()
    FineTune(model=model, device=options.device,
             **reference['finetune']['config']).unlearn(forget, retain)
    report('finetune', model)

    bench.set_seed(options.seed)
    model = benchmark._fresh_source()
    SCRUB(model=model, device=options.device,
          **reference['scrub']['config']).unlearn(forget, retain)
    report('scrub', model)

    bench.set_seed(options.seed)
    model = benchmark._fresh_source()
    MSCRUB(model=model, device=options.device,
           **mmu_results['mmu']['config']).unlearn(forget, retain)
    report('mmu', model)

    print(f'\nAccuracy per nested prefix, CIFAR-10 seed {options.seed} '
          f'(SCRUB cfg {reference["scrub"]["config"]}, MMU cfg {mmu_results["mmu"]["config"]})')
    for split in ('D_f (train)', 'D_f (test)', 'D_r (test)'):
        print(f'\n{split}')
        print(f'{"method":<18}' + ''.join(f'{f"m={m}":>10}' for m in widths))
        for name, values in rows.items():
            print(f'{name:<18}' + ''.join(f'{values[split][m]:>10.2f}' for m in widths))
    (REPOSITORY.parent / 'results/mmu_prefix_leakage' / ('mmu_prefix_leakage_seed%d.json' % options.seed)).write_text(
        json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
