#!/usr/bin/env python3
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

REPO = Path(__file__).resolve().parents[1]
RES = REPO / 'results' / 'classification_benchmark'
OUT = REPO / 'figures' / 'pdf' / 'classification'
DATA = REPO / 'results' / 'figure_data'
sys.path.insert(0, str(REPO / 'mmu'))
from scripts.ulira import attack_metrics, balanced_masks, likelihood_ratio

plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                     'mathtext.default': 'bf', 'xtick.labelsize': 14, 'ytick.labelsize': 14,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
LEGEND = dict(loc='lower center', frameon=True, edgecolor='#444', fancybox=False,
              prop={'weight': 'bold', 'size': 14.5})
DATASETS = [('cifar10', 'CIFAR-10'), ('cifar100', 'CIFAR-100'), ('rti', 'RTI')]
CELLS = [('resnet18', 'class'), ('resnet18', 'subclass'), ('resnet18', 'instance'),
         ('cnn', 'class'), ('cnn', 'subclass'), ('cnn', 'instance')]
MODES = ['class', 'subclass', 'instance']
BACKS = [('resnet18', 'o'), ('cnn', '^')]
NAME = {'cnn': 'CNN', 'resnet18': 'ResNet-18'}
OURS, OURS_LABEL = 'mmu', 'MMU (ours)'
BASELINES = [('baseline', 'Original (no unlearning)', '#555555', 1.5, '--'),
             ('finetune', 'Fine-tune', '#4C8FD0', 1.2, '-'),
             ('ssd', 'SSD', '#F4643C', 1.2, '-'),
             ('badteacher', 'Bad Teacher', '#3FBFA8', 1.2, '-'),
             ('unsir', 'UNSIR', '#F5A623', 1.2, '-'),
             ('scrub', 'SCRUB', '#F191B4', 1.2, '-'),
             ('salun', 'SalUn', '#2E9E3E', 1.2, '-'),
             ('uniclun', 'UniCLUN', '#6B4FA8', 1.2, '-')]
SEEDS = [42, 43, 44]


def cell_dir(ds, bb, mode, seed=42):
    return RES / ds / f'{bb}_adam_{mode}' / f'seed{seed}'


def final(ds, bb, mode, seed=42):
    return json.loads((cell_dir(ds, bb, mode, seed) / 'final.json').read_text())['results']


def tidy(ax):
    ax.grid(color='#E4E4E4', lw=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT.relative_to(REPO) / f'{name}.pdf')


def curve(scores, key, n):
    if f'scores_{key}' not in scores:
        return None
    phi = scores[f'scores_{key}']
    masks = balanced_masks(phi.shape[0], n, 42 * 1000)
    usable = ~np.isnan(phi).any(1)
    pos = likelihood_ratio(phi[usable], masks[usable], scores[f'target_{key}'])
    neg = likelihood_ratio(phi[usable], masks[usable], scores['target_retrain'])
    fpr, tpr, _ = roc_curve(np.r_[np.ones(pos.size), np.zeros(neg.size)], np.r_[pos, neg])
    return fpr, tpr, attack_metrics(pos, neg)


def fig1(ds, label):
    series = [(OURS, OURS_LABEL, 'black', 2.2, '-')] + [(k, l, c, w, s) for k, l, c, w, s in BASELINES]
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.08, wspace=0.12, hspace=0.22)
    for ax, (bb, mode) in zip(axes.ravel(), CELLS):
        d = cell_dir(ds, bb, mode)
        n = json.loads((d / 'ulira.json').read_text())['n_forget']
        scores = np.load(d / 'ulira_scores.npz')
        ax.plot([1e-4, 1], [1e-4, 1], color='#BBB', ls='--', lw=1.1, zorder=1)
        for key, _, colour, lw, ls in series:
            got = curve(scores, key, n)
            if got is None:
                continue
            fpr, tpr, mt = got
            summary.setdefault(f'{bb}/{mode}', {})[key] = mt
            ax.plot(fpr, tpr, color=colour, lw=lw, ls=ls, zorder=3 if key == OURS else 2)
            if key == OURS:
                ax.plot([0.01], [mt['tpr_at_1pct_fpr'] / 100.0], 'o', color='black', ms=6, zorder=4)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1)
        ax.set_title(f'{NAME[bb]} · {mode} (n={n})', fontsize=15.5, pad=7)
        tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel('False positive rate', fontsize=16)
    for ax in axes[:, 0]:
        ax.set_ylabel('True positive rate', fontsize=16)
    handles = [Line2D([], [], color='#BBB', ls='--', lw=2, label='Retrain reference (chance)')] + \
              [Line2D([], [], color=c, lw=w + 1, ls=s, label=l) for _, l, c, w, s in series]
    fig.legend(handles=handles, ncol=4, bbox_to_anchor=(0.5, 0.87), **LEGEND)
    save(fig, f'fig1_ulira_roc_{ds}')
    (DATA / f'fig1_ulira_summary_{ds}.json').write_text(json.dumps(summary, indent=2))


def privacy_rows(ds, bb, mode):
    res = final(ds, bb, mode)
    ul = json.loads((cell_dir(ds, bb, mode) / 'ulira.json').read_text())['results']
    gold = res['retrain']['metrics']
    out = {}
    for name, rec in res.items():
        m, u = rec['metrics'], ul.get(name) or {}
        if 'auc' not in u or name == 'retrain':
            continue
        gap = max(gold['retain_acc'] - m['retain_acc'], gold['test_acc'] - m['test_acc'], 0.0)
        out[name] = {'gap': gap, 'mia': m['mia'], 'auc': u['auc'], 'tpr': max(u['tpr_at_1pct_fpr'], 1e-2)}
    return out


def fig5(ds, label):
    series = [(OURS, OURS_LABEL, 'black')] + [(k, l, c) for k, l, c, _, _ in BASELINES]
    colour = {k: c for k, _, c in series}
    data = {(bb, mode): privacy_rows(ds, bb, mode) for bb, _ in BACKS for mode in MODES}
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(15.2, 10.4))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.985, top=0.80, bottom=0.075, wspace=0.30, hspace=0.36)
    for j, mode in enumerate(MODES):
        ax = fig.add_subplot(gs[0, j])
        pts = []
        for bb, mk in BACKS:
            for name, r in data[(bb, mode)].items():
                if name in colour:
                    ax.scatter([r['gap']], [r['tpr']], marker=mk, s=145 if name == OURS else 78,
                               color=colour[name], zorder=4 if name == OURS else 3, edgecolors='none')
                    pts.append((r['gap'], r['tpr']))
        pts.sort()
        front, best = [], float('inf')
        for x, y in pts:
            if y < best:
                front.append((x, y)); best = y
        if len(front) > 1:
            ax.plot(*zip(*front), color='#999', ls='--', lw=1.2, zorder=2)
        ax.axvline(2.0, color='#E8A0A8', ls='--', lw=1.3, zorder=1)
        ax.set_yscale('log')
        ax.set_title(f'{label} · {mode}', fontsize=16, pad=8)
        ax.set_xlabel('Utility gap (pts)', fontsize=15)
        if j == 0:
            ax.set_ylabel('U-LiRA TPR @ 1% FPR (%, log)', fontsize=15)
        tidy(ax)
    ax = fig.add_subplot(gs[1, :2])
    allpts = [(r['mia'], r['auc'], name, mk, mode) for bb, mk in BACKS for mode in MODES
              for name, r in data[(bb, mode)].items() if name in colour]
    lo = min(min(p[0] for p in allpts), min(p[1] for p in allpts)) - 3
    ax.fill_between([lo, 101], [lo, 101], 101, color='#F6E4E6', zorder=0)
    ax.plot([lo, 101], [lo, 101], color='#444', ls='--', lw=1.3, zorder=2)
    for mia, auc, name, mk, _ in allpts:
        ax.scatter([mia], [auc], marker=mk, s=150 if name == OURS else 62, color=colour[name],
                   zorder=4 if name == OURS else 3,
                   edgecolors='none' if name == OURS else 'white', linewidths=.6)
    ax.set_xlim(lo, 101); ax.set_ylim(lo, 101)
    ax.set_xlabel('Aggregate MIA accuracy (%)', fontsize=15)
    ax.set_ylabel('U-LiRA AUC (%)', fontsize=15)
    ax.set_title(f'{label}, every method-cell pair (n={len(allpts)})', fontsize=16, pad=8)
    ax.annotate('shaded: U-LiRA finds leakage\nthe aggregate MIA misses', xy=(0.50, 0.965),
                xycoords='axes fraction', fontsize=13.5, color='#A2545C', va='top')
    tidy(ax)
    ax = fig.add_subplot(gs[1, 2])
    box = [[r['auc'] - r['mia'] for bb, _ in BACKS for name, r in data[(bb, mode)].items()
            if name in colour] for mode in MODES]
    ax.boxplot(box, patch_artist=True, widths=.55, medianprops=dict(color='#222', lw=1.8),
               boxprops=dict(facecolor='#E4E4E4', edgecolor='#999'), whiskerprops=dict(color='#999'),
               capprops=dict(color='#999'), flierprops=dict(marker='', ls='none'))
    for i, mode in enumerate(MODES, start=1):
        for bb, mk in BACKS:
            for name, r in data[(bb, mode)].items():
                if name in colour:
                    ax.scatter([i + rng.uniform(-.16, .16)], [r['auc'] - r['mia']], marker=mk,
                               s=130 if name == OURS else 34,
                               color=colour[name] if name == OURS else 'none',
                               edgecolors=colour[name], linewidths=1.1, zorder=4 if name == OURS else 3)
    ax.axhline(0, color='#444', ls='--', lw=1.2)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(MODES)
    ax.set_xlabel('Forget mode', fontsize=15)
    ax.set_ylabel('U-LiRA AUC − MIA (pts)', fontsize=15)
    ax.set_title('Under-reporting by mode', fontsize=16, pad=8)
    tidy(ax)
    handles = [Line2D([], [], marker='o', ls='', ms=13 if k == OURS else 9, color=c, label=l)
               for k, l, c in series]
    handles += [Line2D([], [], marker='o', ls='', ms=9, color='#AAA', label='ResNet-18 (circle)'),
                Line2D([], [], marker='^', ls='', ms=9, color='#AAA', label='CNN (triangle)'),
                Line2D([], [], color='#999', ls='--', lw=1.2, label='Pareto frontier'),
                Line2D([], [], color='#E8A0A8', ls='--', lw=1.3, label='2-pt utility tolerance'),
                Line2D([], [], color='#444', ls='--', lw=1.3, label='perfect agreement')]
    fig.legend(handles=handles, ncol=5, bbox_to_anchor=(0.5, 0.84), **LEGEND)
    save(fig, f'fig5_privacy_combined_{ds}')
    diff = [a - m for m, a, *_ in allpts]
    by_mode = {mode: [a - m for m, a, _, _, md in allpts if md == mode] for mode in MODES}
    stats = dict(n_pairs=len(allpts), above_diagonal=sum(d > 0 for d in diff),
                 median_gap=st.median(diff), min_gap=min(diff), max_gap=max(diff),
                 median_gap_by_mode={k: st.median(v) for k, v in by_mode.items()},
                 mmu={f'{bb}/{mode}': data[(bb, mode)].get(OURS) for bb, _ in BACKS for mode in MODES})
    (DATA / f'fig5_privacy_stats_{ds}.json').write_text(json.dumps(stats, indent=2))


def fig4():
    order = ['retrain', OURS, 'finetune', 'badteacher', 'uniclun', 'salun', 'ssd', 'scrub', 'unsir']
    names = {'retrain': 'Retrain', OURS: 'MMU', **{k: l for k, l, *_ in BASELINES}}
    col = {'retrain': '#6b8f3a', OURS: '#e34948', **{k: c for k, _, c, *_ in BASELINES}}
    agg = {}
    for ds, label in DATASETS:
        times = {}
        for p in RES.glob(f'{ds}/*/seed*/final.json'):
            for m, rec in json.loads(p.read_text())['results'].items():
                times.setdefault(m, []).append(rec['metrics']['unlearn_time'])
        agg[label] = {m: (st.mean(v), min(v), max(v), len(v)) for m, v in times.items()}
    width = 4.3 * len(agg) + 1.1
    fig, axes = plt.subplots(1, len(agg), figsize=(width, 6.2), sharey=True)
    fig.subplots_adjust(left=1.82 / width, right=.99, bottom=.12, top=.87, wspace=.1)
    ypos = np.arange(len(order))[::-1]
    for ax, (label, times) in zip(axes, agg.items()):
        for y, m in zip(ypos, order):
            if m not in times:
                continue
            mu, lo, hi, _ = times[m]
            ax.plot([2, mu], [y, y], color='#e6e5e1', lw=2.0, zorder=2, solid_capstyle='butt')
            ax.plot([lo, hi], [y, y], color=col[m], lw=3.0, alpha=.55, zorder=3)
            ax.plot([lo, hi], [y, y], '|', color=col[m], ms=9, mew=2.0, alpha=.85, zorder=3)
            ax.plot([mu], [y], 'o', ms=15 if m == OURS else 12.5, color=col[m], mec='white', mew=2.0, zorder=5)
            ax.text(hi * 1.3, y, f'{mu:.0f}s', va='center', fontsize=13)
        ax.set_xscale('log'); ax.set_xlim(2, 5000); ax.set_ylim(-.5, len(order) - .5)
        ax.grid(True, axis='x', color='#e6e5e1'); ax.set_axisbelow(True)
        ax.set_xlabel('Unlearning time (s, log)', fontsize=15)
        ax.set_title(f'{label} ({times["mmu"][3]} cell-seeds)', fontsize=15, pad=6)
        ax.tick_params(axis='y', length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([names[m] for m in order], fontsize=15)
    handles = [Line2D([], [], marker='o', ls='', mfc=col[OURS], mec='white', ms=14, label=OURS_LABEL),
               Line2D([], [], marker='o', ls='', mfc=col['retrain'], mec='white', ms=12, label='Retrain oracle'),
               Line2D([], [], color='#3a3936', lw=3.0, marker='|', ms=9, mew=2, alpha=.6, label='min–max across cell-seeds')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5 + .9 / width, .925), ncol=3,
               frameon=True, edgecolor='#cfcec9', prop={'weight': 'bold', 'size': 14})
    save(fig, 'fig4_unlearning_time')
    (DATA / 'fig4_unlearning_time.json').write_text(json.dumps(
        {lab: {m: dict(mean=v[0], min=v[1], max=v[2], n=v[3]) for m, v in t.items()}
         for lab, t in agg.items()}, indent=2))


def fig6(ds, label):
    series = [('retrain', 'Retrain', '#6b8f3a', 2.2, '-'), (OURS, OURS_LABEL, 'black', 2.4, '-')] + BASELINES
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.08, wspace=0.10, hspace=0.25)
    table = {}
    for ax, (bb, mode) in zip(axes.ravel(), CELLS):
        per_seed = [final(ds, bb, mode, s) for s in SEEDS]
        for key, _, colour, lw, ls in series:
            curves = [[r[key]['metrics']['forget_acc']] + r[key]['metrics']['relearn_forget_acc']
                      for r in per_seed if key in r]
            if not curves:
                continue
            c = np.array(curves)
            mean, sd = c.mean(0), c.std(0, ddof=1) if len(c) > 1 else np.zeros(c.shape[1])
            x = np.arange(len(mean))
            ax.plot(x, mean, color=colour, lw=lw, ls=ls, zorder=4 if key == OURS else 3)
            if key in (OURS, 'retrain'):
                ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=.12, lw=0)
            table.setdefault(f'{bb}/{mode}', {})[key] = [round(v, 2) for v in mean]
        ax.set_title(f'{NAME[bb]} · {mode}', fontsize=15.5, pad=7)
        ax.set_xticks(range(6))
        tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel('Relearn epochs on D$_f$', fontsize=15)
    for ax in axes[:, 0]:
        ax.set_ylabel('D$_f$ accuracy (%)', fontsize=15)
    handles = [Line2D([], [], color=c, lw=w + 1, ls=s, label=l) for _, l, c, w, s in series]
    fig.legend(handles=handles, ncol=5, bbox_to_anchor=(0.5, 0.86), **LEGEND)
    save(fig, f'fig6_recovery_{ds}')
    (DATA / f'fig6_recovery_{ds}.json').write_text(json.dumps(table, indent=2))


if __name__ == '__main__':
    DATA.mkdir(parents=True, exist_ok=True)
    for ds, label in DATASETS:
        fig1(ds, label)
        fig5(ds, label)
        fig6(ds, label)
    fig4()
