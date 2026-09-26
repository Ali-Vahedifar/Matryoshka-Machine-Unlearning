#!/usr/bin/env python3
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / 'results' / 'classification_benchmark'
OUT = REPO / 'results' / 'tables'

DATASETS = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'rti': 'RTI'}
CONTROLS = ('baseline', 'retrain')
EXCLUDED = ('finetune',)
ORDER = ['retrain', 'baseline', 'finetune', 'amnesiac', 'badteacher', 'scrub', 'ssd',
         'salun', 'unsir', 'uniclun', 'mmu']
LABEL = {'retrain': 'Retrain (oracle)', 'baseline': 'Original model',
         'finetune': 'Fine-tune', 'amnesiac': 'Amnesiac',
         'badteacher': 'Bad Teacher', 'scrub': 'SCRUB', 'ssd': 'SSD', 'salun': 'SalUn',
         'unsir': 'UNSIR', 'uniclun': 'UniCLUN', 'mmu': 'MMU (ours)'}
SETUPS = [('fpecnn', 'class'), ('fpecnn', 'subclass'), ('fpecnn', 'instance'),
          ('resnet18', 'class'), ('resnet18', 'subclass'), ('resnet18', 'instance')]
BACK = {'fpecnn': 'CNN', 'resnet18': 'ResNet-18'}
NUM = ['forget_acc', 'retain_acc', 'test_acc', 'mia', 'forget_gap', 'retain_drop',
       'test_drop', 'relearn_1ep', 'relearn_last', 'ulira_auc', 'time_s']


def per_seed_rows(ds):
    out = []
    for path in sorted((RES / ds).glob('*/seed*/final.json')):
        data = json.loads(path.read_text())
        pr, results = data['protocol'], data['results']
        gold = results['retrain']['metrics']
        ulira_path = path.parent / 'ulira.json'
        ulira = json.loads(ulira_path.read_text())['results'] if ulira_path.exists() else {}
        for name, record in results.items():
            m = record['metrics']
            curve = m.get('relearn_forget_acc') or []
            out.append({
                'cell': path.parts[-3], 'backbone': pr['backbone'], 'mode': pr['forget_mode'],
                'seed': pr['seed'], 'method': name, 'label': LABEL.get(name, name),
                'feasible': record.get('feasible'),
                'forget_acc': m['forget_acc'], 'retain_acc': m['retain_acc'],
                'test_acc': m['test_acc'], 'mia': m['mia'], 'mia_auc': m.get('mia_auc'),
                'forget_gap': abs(m['forget_acc'] - gold['forget_acc']),
                'retain_drop': gold['retain_acc'] - m['retain_acc'],
                'test_drop': gold['test_acc'] - m['test_acc'],
                'relearn_1ep': curve[0] if curve else None,
                'relearn_last': curve[-1] if curve else None,
                'ulira_auc': ulira.get(name, {}).get('auc'),
                'ulira_tpr1': ulira.get(name, {}).get('tpr_at_1pct_fpr'),
                'time_s': m['unlearn_time'],
            })
    return out


def aggregate(per_seed):
    groups = defaultdict(list)
    for r in per_seed:
        groups[(r['backbone'], r['mode'], r['method'])].append(r)
    out = []
    for (backbone, mode, method), rs in sorted(groups.items()):
        row = {'backbone': backbone, 'mode': mode, 'method': method,
               'label': LABEL.get(method, method), 'n_seeds': len(rs),
               'feasible_seeds': sum(1 for r in rs if r['feasible'])}
        for key in NUM:
            vals = [r[key] for r in rs if r[key] is not None]
            row[f'{key}_mean'] = statistics.mean(vals) if vals else None
            row[f'{key}_sd'] = statistics.pstdev(vals) if len(vals) > 1 else None
        out.append(row)
    return out


def write_csv(path, data):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def fmt(row, key):
    m, s = row.get(f'{key}_mean'), row.get(f'{key}_sd')
    if m is None:
        return '—'
    return f'{m:.1f}' if s is None else f'{m:.1f} ± {s:.1f}'


def markdown(ds, agg):
    lines = [f'# {DATASETS[ds]}', '',
             'Mean ± SD over seeds 42, 43, 44. `feasible` counts seeds whose selected configuration '
             'kept retain and test accuracy within 2 points of Retrain. Relearn = D_f accuracy after '
             '1 / 5 Adam epochs on D_f. U-LiRA AUC is seed 42 (50 = indistinguishable from Retrain).', '']
    for backbone in ('fpecnn', 'resnet18'):
        for mode in ('class', 'subclass', 'instance'):
            block = [r for r in agg if r['backbone'] == backbone and r['mode'] == mode]
            if not block:
                continue
            block.sort(key=lambda r: (r['method'] == 'baseline', r['method'] != 'retrain',
                                      r['forget_gap_mean'] if r['forget_gap_mean'] is not None else 1e9))
            lines += [f'## {BACK[backbone]} / {mode}', '',
                      '| Method | seeds | feasible | D_f acc | D_r acc | Test acc | |ΔD_f| | ΔD_r | MIA | relearn 1ep | relearn 5ep | U-LiRA AUC | time (s) |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in block:
                lines.append(
                    f"| {r['label']} | {r['n_seeds']} | {r['feasible_seeds']} | {fmt(r, 'forget_acc')} | "
                    f"{fmt(r, 'retain_acc')} | {fmt(r, 'test_acc')} | {fmt(r, 'forget_gap')} | "
                    f"{fmt(r, 'retain_drop')} | {fmt(r, 'mia')} | {fmt(r, 'relearn_1ep')} | "
                    f"{fmt(r, 'relearn_last')} | {fmt(r, 'ulira_auc')} | {fmt(r, 'time_s')} |")
            lines.append('')
    return '\n'.join(lines)


def main():
    for ds in DATASETS:
        out = OUT / ds
        out.mkdir(parents=True, exist_ok=True)
        per_seed = per_seed_rows(ds)
        agg = aggregate(per_seed)
        write_csv(out / 'per_seed.csv', per_seed)
        write_csv(out / 'mean_3seed.csv', agg)
        (out / 'summary.md').write_text(markdown(ds, agg))
        print(f'{DATASETS[ds]}: {len(per_seed)} rows -> {out.relative_to(REPO)}')


if __name__ == '__main__':
    main()
