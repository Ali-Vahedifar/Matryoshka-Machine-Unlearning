#!/usr/bin/env python3
import json
import re
import statistics as st
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CB = REPO / 'results' / 'classification_benchmark'
SP = REPO / 'generation' / 'salun_protocol'
OUT = REPO / 'results' / 'MMU_results.xlsx'

LABEL = {'baseline': 'Original', 'retrain': 'Retrain', 'finetune': 'Fine-tuning',
         'badteacher': 'Bad Teacher', 'amnesiac': 'Amnesiac', 'unsir': 'UNSIR', 'ssd': 'SSD',
         'salun': 'SalUn', 'uniclun': 'UniCLUN', 'scrub': 'SCRUB', 'mmu': 'MMU (ours)'}
DS = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'rti': 'RTI'}
BB = {'cnn': 'CNN', 'resnet18': 'ResNet-18'}
KEYS = ['dataset', 'backbone', 'mode', 'method']
METRICS = ['abs_dDf', 'Df_acc', 'Dr_acc', 'test_acc', 'MIA', 'MIA_AUC', 'output_KL', 'param_KL',
           'unlearn_time_s'] + [f'relearn_ep{i}' for i in range(1, 6)]


def cells():
    for p in sorted(CB.glob('*/*/seed*/final.json')):
        ds, cell, seed = p.parts[-4], p.parts[-3], int(p.parts[-2][4:])
        bb, _, mode = cell.split('_')
        yield p.parent, DS[ds], BB[bb], mode, seed


def ordered(df):
    if 'method' in df:
        df['method'] = pd.Categorical(df['method'], list(LABEL.values()), ordered=True)
    df['mode'] = pd.Categorical(df['mode'], ['class', 'subclass', 'instance'], ordered=True)
    return df.sort_values([c for c in ['dataset', 'backbone', 'mode', 'method', 'seed'] if c in df])


def per_seed():
    rows = []
    for d, ds, bb, mode, seed in cells():
        res = json.loads((d / 'final.json').read_text())['results']
        ref = res['retrain']['metrics']['forget_acc']
        for m, rec in res.items():
            x = rec['metrics']
            rows.append(dict(dataset=ds, backbone=bb, mode=mode, seed=seed, method=LABEL[m],
                             abs_dDf=abs(x['forget_acc'] - ref), Df_acc=x['forget_acc'],
                             Dr_acc=x['retain_acc'], test_acc=x['test_acc'], MIA=x['mia'],
                             MIA_AUC=x.get('mia_auc'), output_KL=x['output_kl_divergence'],
                             param_KL=x['kl_divergence'], unlearn_time_s=x['unlearn_time'],
                             **{f'relearn_ep{i}': v for i, v in enumerate(x.get('relearn_forget_acc') or [], 1)},
                             feasible=rec.get('feasible'), selected_config=json.dumps(rec.get('config') or {}, sort_keys=True),
                             access=rec.get('access'), source_sha256=rec.get('source_sha256')))
    return ordered(pd.DataFrame(rows))


def summary(seeds):
    g = seeds.groupby(KEYS, observed=True)
    out = g[METRICS].agg(['mean', 'std'])
    out.columns = [f'{m}_{s}' for m, s in out.columns]
    out.insert(0, 'n_seeds', g.size())
    out.insert(1, 'feasible_seeds', g['feasible'].apply(lambda s: int((s == True).sum()) if s.notna().any() else None))
    return out.reset_index()


def formatted(summ):
    f = summ[KEYS + ['n_seeds']].copy()
    for m, digits in [('abs_dDf', 2), ('Dr_acc', 2), ('test_acc', 2), ('MIA', 2), ('output_KL', 3),
                      ('param_KL', 4), ('unlearn_time_s', 1), ('relearn_ep1', 2), ('relearn_ep5', 2)]:
        f[m] = [f'{a:.{digits}f} ± {b:.{digits}f}' if pd.notna(a) else '' for a, b in
                zip(summ[f'{m}_mean'], summ[f'{m}_std'].fillna(0))]
    return f


def ulira():
    rows = []
    for d, ds, bb, mode, seed in cells():
        if not (d / 'ulira.json').exists():
            continue
        u = json.loads((d / 'ulira.json').read_text())
        for m, r in u['results'].items():
            rows.append(dict(dataset=ds, backbone=bb, mode=mode, seed=seed, method=LABEL[m],
                             n_forget=u['n_forget'], shadows=u['shadows'], AUC=r.get('auc'),
                             TPR_at_1pct_FPR=r.get('tpr_at_1pct_fpr'), TPR_at_0p1pct_FPR=r.get('tpr_at_0.1pct_fpr'),
                             balanced_acc=r.get('balanced_acc'), finite_shadows=r.get('finite_shadows'),
                             note=r.get('note') or r.get('error')))
    return ordered(pd.DataFrame(rows))


def training():
    rows = []
    for d, ds, bb, mode, seed in cells():
        if not (d / 'manifest.json').exists():
            continue
        m = json.loads((d / 'manifest.json').read_text())
        s, r = m.get('source_training', {}), m.get('retrain_training', {})
        rows.append(dict(dataset=ds, backbone=bb, mode=mode, seed=seed,
                         source_best_epoch=s.get('best_epoch'), source_epochs_ran=s.get('epochs_ran'),
                         source_best_val_acc=s.get('best_val_acc'), source_train_time_s=m.get('source_train_time'),
                         retrain_best_epoch=r.get('best_epoch'), retrain_epochs_ran=r.get('epochs_ran'),
                         retrain_best_val_acc=r.get('best_val_acc'), retrain_time_s=m.get('retrain_time')))
    return ordered(pd.DataFrame(rows))


def salun_protocol():
    methods = ['retrain', 'FT', 'RL', 'GA', 'IU', 'BE', 'BS', 'l1_sparse', 'SalUn', 'SalUn_soft', 'MMU', 'MMU2T']
    summ, trials = [], []
    for ratio in [10, 50]:
        data = {m: [json.loads(p.read_text()) for s in range(2, 12)
                    if (p := SP / f'classification/seed{s}/forget{ratio}/{m}.json').exists()] for m in methods}
        refs = {r['seed']: r for r in data['retrain']}
        for m, recs in data.items():
            recs = [r for r in recs if r['seed'] in refs]
            row = dict(forget_pct=ratio, method=m, trials=len(recs))
            gaps = []
            for k, name in [('UA', 'Df_acc'), ('RA', 'Dr_acc'), ('TA', 'test_acc'), ('MIA', 'MIA')]:
                val = lambda r: 100 - r[k] if k == 'UA' else r[k]
                v = [val(r) for r in recs]
                row[f'{name}_mean'], row[f'{name}_std'] = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
                row[f'{name}_gap'] = abs(st.mean(v) - st.mean(val(refs[r['seed']]) for r in recs))
                gaps.append(row[f'{name}_gap'])
            row['avg_gap'] = st.mean(gaps)
            row['RTE_min'] = st.mean(r['RTE_min'] for r in recs)
            summ.append(row)
            for r in recs:
                trials.append(dict(forget_pct=ratio, method=m, seed=r['seed'], Df_acc=100 - r['UA'], Dr_acc=r['RA'],
                                   test_acc=r['TA'], MIA=r['MIA'], RTE_min=r['RTE_min'],
                                   hyperparameters=json.dumps(r.get('hyperparameters', {}), sort_keys=True)))
    return pd.DataFrame(summ), pd.DataFrame(trials)


def sd_seed_study():
    study = json.loads((SP / 'sd_concept_one_seed/seed_study.json').read_text())
    rows = [dict(variant=k, seed=int(s), detected_of_20=v['detected'], retain_CLIP=v['retain_clip'],
                 detected_prompt_ids=','.join(map(str, v['ids']))) for k, per in study.items() for s, v in per.items()]
    seeds = pd.DataFrame(rows)
    g = seeds.groupby('variant', sort=False)
    summ = pd.DataFrame({'n_seeds': g.size(), 'detected_mean': g.detected_of_20.mean(), 'detected_std': g.detected_of_20.std(),
                         'retain_CLIP_mean': g.retain_CLIP.mean(), 'retain_CLIP_std': g.retain_CLIP.std()}).reset_index()
    return summ, seeds


def prefix_leakage():
    rows = []
    for p in sorted((REPO / 'results/mmu_prefix_leakage').glob('mmu_prefix_leakage_seed*.json')):
        seed = int(re.search(r'seed(\d+)', p.name).group(1))
        for method, splits in json.loads(p.read_text()).items():
            for split, widths in splits.items():
                for w, acc in widths.items():
                    rows.append(dict(seed=seed, method=method, split=split, width=int(w), accuracy=acc))
    return pd.DataFrame(rows)


def variants(path):
    rows = []
    for cell, per in json.loads(path.read_text()).items():
        for vid, r in per.items():
            rows.append(dict(cell=cell, id=vid, name=r.get('name'), seconds=r.get('seconds'),
                             **{k: v for k, v in r.get('metrics', {}).items()},
                             config=json.dumps(r.get('config') or {}, sort_keys=True)))
    return pd.DataFrame(rows)


def md_tables(path):
    blocks, heading, buf = [], path.stem, []
    for line in path.read_text().splitlines() + ['']:
        if line.startswith('#'):
            heading = line.lstrip('# ').strip()
        if line.startswith('|'):
            buf.append([c.strip() for c in line.strip('|').split('|')])
            continue
        if buf:
            head, body = buf[0], [r for r in buf[2:] if len(r) == len(buf[0])]
            blocks.append((f'{path.relative_to(REPO)} — {heading}', pd.DataFrame(body, columns=head)))
            buf = []
    return blocks


NOTES = [
    ('What', 'Final results of the MMU paper, built by tools/build_results_workbook.py from the committed JSON records.'),
    ('Classification benchmark', 'results/classification_benchmark/<dataset>/<backbone>_adam_<mode>/seed<k>/final.json; '
                                 'protocol in benchmarks/PROTOCOL.md. Seeds 42/43/44; U-LiRA on seed 42 with 16 shadows.'),
    ('|ΔDf|', 'Absolute forget-accuracy gap to the Retrain model of the same seed, then mean ± sd over seeds.'),
    ('MIA / MIA_AUC', 'Aggregate membership-inference attack accuracy / AUC (independent calibration subsets).'),
    ('output_KL / param_KL', 'Predictive KL and parameter-distribution KL to Retrain.'),
    ('unlearn_time_s', 'Wall-clock seconds; Retrain = its training time. Three cells ran in parallel on one GPU, '
                       'so times include contention.'),
    ('relearn_ep1..5', 'D_f accuracy after each of 5 relearn epochs (Adam, lr 1e-4) on D_f.'),
    ('feasible_seeds', 'Seeds on which the selected configuration met the 2-point utility tolerance vs Retrain.'),
    ('RTI', '20-class dataset; settings in benchmarks/PROTOCOL.md.'),
    ('SalUn protocol', 'generation/salun_protocol/classification: CIFAR-10 ResNet-18, 10%/50% random forgetting, '
                       'seeds 2-11, official SalUn release for FT/RL/GA/IU/BE/BS/l1-sparse/SalUn. gap = |mean(method) - mean(paired Retrain)|.'),
    ('SD nudity', 'generation/salun_protocol/sd_concept_one_seed: SD v1.4, NudeNet 3.4.2 at 0.6, 20 forget + 20 retain prompts.'),
    ('DDPM', 'generation/diffusion_mmu: CIFAR-10 class-conditional DDPM, forget class airplane.'),
]


def main():
    seeds = per_seed()
    summ = summary(seeds)
    sal_summ, sal_trials = salun_protocol()
    sd_summ, sd_seeds = sd_seed_study()
    sheets = {
        'Notes': pd.DataFrame(NOTES, columns=['item', 'description']),
        'Classif_mean_std': summ,
        'Classif_formatted': formatted(summ),
        'Classif_per_seed': seeds,
        'ULiRA_seed42': ulira(),
        'Source_Retrain_training': training(),
        'SalUn_protocol_summary': sal_summ,
        'SalUn_protocol_trials': sal_trials,
        'SD_nudity_3seeds': sd_summ,
        'SD_nudity_per_seed': sd_seeds,
        'MMU_prefix_leakage': prefix_leakage(),
        'MMU_variants_3seed': variants(SP / 'mmu_variants_3seed/results.json'),
        'MMU_margin_lr5e-3': variants(SP / 'mmu_margin_lr005/results.json'),
    }
    reports = {
        'DDPM_CIFAR10': [REPO / 'generation/diffusion_mmu/results/full_redirect_20260907_comparison/comparison.md'],
        'SD_pilot_seed42': [SP / 'sd_concept_one_seed/report.md'],
        'MMU2T_strengthening': [SP / 'mmu2t_strengthening/report.md'],
    }
    with pd.ExcelWriter(OUT, engine='openpyxl') as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)
        for name, paths in reports.items():
            row = 0
            for p in paths:
                for title, df in md_tables(p):
                    pd.DataFrame([[title]]).to_excel(xl, sheet_name=name, index=False, header=False, startrow=row)
                    df.to_excel(xl, sheet_name=name, index=False, startrow=row + 1)
                    row += len(df) + 3
        for ws in xl.book.worksheets:
            ws.freeze_panes = 'A2'
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    print('wrote', OUT.relative_to(REPO), {k: len(v) for k, v in sheets.items()})


if __name__ == '__main__':
    main()
