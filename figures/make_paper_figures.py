import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
MU = REPO
ROOT = REPO / 'figures' / 'pdf' / 'paper'
DATA = REPO / 'results' / 'figure_data'
SEEDS = [42, 43, 44]

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'axes.edgecolor': MUTED, 'axes.linewidth': .8,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.labelcolor': INK,
    'legend.frameon': False, 'figure.facecolor': 'white',
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'grid.alpha': .9,
})


def save(fig, stem):
    ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f'{stem}.pdf', bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
    print('wrote', stem)


def tidy(ax):
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def fig_prefix_depth():
    rows = [('original', 'Original', MUTED, '--', 'o'),
            ('retrain (oracle)', 'Retrain (oracle)', INK, '--', 's'),
            ('finetune', 'Fine-tune', BLUE, '-', '^'),
            ('scrub', 'SCRUB', ORANGE, '-', 'v'),
            ('mmu', 'MMU', AQUA, '-', 'D')]
    per_seed = [json.loads((MU / f'results/mmu_prefix_leakage/mmu_prefix_leakage_seed{s}.json').read_text())
                for s in SEEDS]
    widths = sorted(int(w) for w in per_seed[0]['original']['D_f (train)'])

    table = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, key, title in [(axes[0], 'D_f (train)', 'Forgotten data'),
                           (axes[1], 'D_r (test)', 'Retained data')]:
        for name, label, colour, style, marker in rows:
            vals = np.array([[d[name][key][str(w)] for w in widths] for d in per_seed])
            mean, sd = vals.mean(0), vals.std(0, ddof=1)
            ax.plot(widths, mean, style, color=colour, marker=marker, markersize=4.5,
                    linewidth=1.8, label=label, zorder=3)
            ax.fill_between(widths, mean - sd, mean + sd, color=colour, alpha=.13,
                            linewidth=0, zorder=2)
            table.setdefault(label, {})[f'{title} m={widths}'] = (
                [round(v, 2) for v in mean], [round(v, 2) for v in sd])
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels(widths)
        ax.set_xlabel('nesting width $m$')
        ax.set_title(title, loc='left', color=INK)
        ax.set_ylim(-4, 104)
        tidy(ax)
    axes[0].set_ylabel('accuracy (%)')
    axes[0].legend(loc='center left', fontsize=8, labelcolor=INK)
    save(fig, 'fig1_prefix_depth')
    with (DATA / 'fig1_prefix_depth.csv').open('w') as f:
        f.write('panel,method,width,mean,sd\n')
        for name, label, *_ in rows:
            for key, title in [('D_f (train)', 'forgotten'), ('D_r (test)', 'retained')]:
                vals = np.array([[d[name][key][str(w)] for w in widths] for d in per_seed])
                for w, m_, s_ in zip(widths, vals.mean(0), vals.std(0, ddof=1)):
                    f.write(f'{title},{label},{w},{m_:.2f},{s_:.2f}\n')


def balanced_masks(shadows, n, seed):
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((shadows, n)), axis=0) < shadows // 2


def likelihood_ratio(shadow_scores, in_mask, target_scores):
    in_count, out_count = in_mask.sum(0), (~in_mask).sum(0)
    mean_in = np.where(in_mask, shadow_scores, 0).sum(0) / in_count
    mean_out = np.where(~in_mask, shadow_scores, 0).sum(0) / out_count
    var_in = np.mean(((shadow_scores - mean_in) ** 2)[in_mask]) + 1e-6
    var_out = np.mean(((shadow_scores - mean_out) ** 2)[~in_mask]) + 1e-6

    def log_pdf(x, mean, var):
        return -.5 * ((x - mean) ** 2 / var + np.log(2 * np.pi * var))

    return log_pdf(target_scores, mean_in, var_in) - log_pdf(target_scores, mean_out, var_out)


def fig_ulira_roc(cell='resnet18_adam_instance', seed=42):
    z = np.load(MU / f'results/classification_benchmark/cifar10/{cell}/seed{seed}/ulira_scores.npz')
    reference = z['target_retrain']
    shown = [('baseline', 'Original', MUTED, '--'),
             ('finetune', 'Fine-tune', BLUE, '-'),
             ('scrub', 'SCRUB', ORANGE, '-'),
             ('salun', 'SalUn', YELLOW, '-'),
             ('mmu', 'MMU', AQUA, '-')]
    n = len(reference)
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.plot([1e-4, 1], [1e-4, 1], color=GRID, linewidth=1, zorder=1)
    ax.annotate('chance', (1.4e-2, 1.0e-2), color=MUTED, fontsize=7.5, rotation=32)
    summary = {}
    for name, label, colour, style in shown:
        if f'scores_{name}' not in z.files:
            continue
        phi = z[f'scores_{name}']
        masks = balanced_masks(len(phi), n, seed * 1000)
        usable = ~np.isnan(phi).any(1)
        phi, masks = phi[usable], masks[usable]
        pos = likelihood_ratio(phi, masks, z[f'target_{name}'])
        neg = likelihood_ratio(phi, masks, reference)
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        scores = np.concatenate([pos, neg])
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = 100 * roc_auc_score(labels, scores)
        at001 = 100 * float(np.interp(.001, fpr, tpr))
        summary[label] = (auc, at001)
        ax.plot(np.clip(fpr, 1e-4, 1), np.clip(tpr, 1e-4, 1), style, color=colour,
                linewidth=1.8, label=f'{label}  (AUC {auc:.1f})', zorder=3)
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlim(1e-3, 1), ax.set_ylim(1e-3, 1)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('U-LiRA, forgotten examples', loc='left', color=INK)
    ax.legend(loc='lower right', fontsize=7.5, labelcolor=INK)
    tidy(ax)
    save(fig, 'fig2_ulira_roc')
    (DATA / 'fig2_ulira_summary.json').write_text(json.dumps(
        {k: {'auc': v[0], 'tpr_at_0.1pct_fpr': v[1]} for k, v in summary.items()}, indent=2))


def fig_saliency_null():
    study = json.loads((MU / 'generation/salun_protocol/sd_concept_one_seed/seed_study.json').read_text())
    seeds = [str(s) for s in SEEDS]
    pairs = [('salun_port', 'SalUn', BLUE), ('random_mask', 'Random mask', ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    fig.subplots_adjust(wspace=.42)
    for ax, key, ylabel, title in [
            (axes[0], 'detected', 'detections / 20', 'Forgotten prompts'),
            (axes[1], 'retain_clip', 'retained CLIP cosine', 'Retained prompts')]:
        for j, (name, label, colour) in enumerate(pairs):
            y = [study[name][s][key] for s in seeds]
            ax.plot([j] * len(y), y, 'o', color=colour, markersize=7,
                    markeredgecolor='white', markeredgewidth=1.4, zorder=3)
        for i, s in enumerate(seeds):
            ax.plot([0, 1], [study[pairs[0][0]][s][key], study[pairs[1][0]][s][key]],
                    '-', color=MUTED, linewidth=.9, alpha=.55, zorder=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([p[1] for p in pairs])
        ax.set_xlim(-.45, 1.45)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', color=INK)
        tidy(ax)
    axes[0].set_ylim(-.6, 2.2)
    axes[0].annotate('0 / 20 on every seed,\nboth methods', (.5, 1.15), ha='center',
                     color=MUTED, fontsize=8)
    save(fig, 'fig3_saliency_null')


def fig_margin_dial():
    tune = MU / 'generation/salun_protocol/classification/seed1/forget10/tuning'
    ref = json.loads((tune / 'retrain.json').read_text())
    points = []
    for p in tune.glob('MMU2T_candidate*.json'):
        r = json.loads(p.read_text())
        h = r['hyperparameters']
        points.append((h['lr'], h['alpha'],
                       abs(r['UA'] - ref['UA']), abs(r['MIA'] - ref['MIA']),
                       abs(r['TA'] - ref['TA']),
                       sum(abs(r[k] - ref[k]) for k in ['UA', 'RA', 'TA', 'MIA']) / 4))
    lrs = sorted({p[0] for p in points})
    colours = dict(zip(lrs, [BLUE, ORANGE, AQUA]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for lr in lrs:
        sub = sorted([p for p in points if p[0] == lr], key=lambda p: p[1])
        m = [p[1] for p in sub]
        axes[0].plot(m, [p[5] for p in sub], '-o', color=colours[lr], markersize=5,
                     linewidth=1.8, label=f'lr {lr:g}', zorder=3)
        axes[1].plot(m, [p[3] for p in sub], '-o', color=colours[lr], markersize=5,
                     linewidth=1.8, zorder=3)
    best = min(points, key=lambda p: p[5])
    axes[0].plot([best[1]], [best[5]], 'o', markersize=11, markerfacecolor='none',
                 markeredgecolor=INK, markeredgewidth=1.4, zorder=4)
    axes[0].annotate('selected', (best[1], best[5]), xytext=(6, 9),
                     textcoords='offset points', color=INK, fontsize=8)
    for ax, ylabel, title in [(axes[0], 'mean |gap| to Retrain (pts)', 'Overall'),
                              (axes[1], '|MIA gap| to Retrain (pts)', 'Privacy')]:
        ax.set_xscale('log')
        ax.set_xticks(sorted({p[1] for p in points}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel('bad-teacher margin $\\mu$')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', color=INK)
        tidy(ax)
    axes[0].axhline(4.74, color=MUTED, linestyle=':', linewidth=1.2, zorder=1)
    axes[0].annotate('MMU (no margin)', (.055, 4.9), color=MUTED, fontsize=8)
    axes[0].legend(loc='upper right', fontsize=8, labelcolor=INK)
    save(fig, 'fig4_margin_dial')


def fig_cross_domain():
    cls = MU / 'generation/salun_protocol/classification'
    names = {'FT': 'FT', 'RL': 'RL', 'GA': 'GA', 'IU': 'IU', 'BE': 'BE', 'BS': 'BS',
             'l1_sparse': '$\\ell_1$', 'SalUn': 'SalUn', 'SalUn_soft': 'SalUn-soft',
             'MMU': 'MMU', 'MMU2T': 'MMU-2T'}
    agg = {}
    for m in names:
        fg, ta = [], []
        for s in range(2, 12):
            p, r = cls / f'seed{s}/forget10/{m}.json', cls / f'seed{s}/forget10/retrain.json'
            if not p.exists():
                continue
            a, b = json.loads(p.read_text()), json.loads(r.read_text())
            fg.append(abs(a['UA'] - b['UA']))
            ta.append(a['TA'] - b['TA'])
        if fg:
            agg[m] = (np.mean(fg), np.mean(ta))

    gen = json.loads((MU / 'generation/salun_protocol/sd_concept_one_seed/seed_study.json').read_text())
    gnames = {'salun_port': 'SalUn', 'random_mask': 'Random mask',
              'mmu_amp10': 'MMU (redirect)', 'mmu_2t_hi': 'MMU-2T', 'mmu_nested': 'MMU nested'}

    offsets = {'MMU': (7, 4), 'MMU2T': (8, -2), 'SalUn': (-30, -3),
               'SalUn_soft': (-42, -9), 'l1_sparse': (7, -2), 'FT': (-20, 4)}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    fig.subplots_adjust(wspace=.34)
    ax = axes[0]
    for m, (x, y) in agg.items():
        ours = m.startswith('MMU')
        ax.plot(x, y, 'o', color=AQUA if ours else MUTED, markersize=8 if ours else 5.5,
                markeredgecolor='white', markeredgewidth=1.2, zorder=4 if ours else 3)
        if m in offsets:
            ax.annotate(names[m], (x, y), xytext=offsets[m], textcoords='offset points',
                        fontsize=7.5, color=INK if ours else MUTED)
    ax.plot(0, 0, '*', color=INK, markersize=13, zorder=5)
    ax.annotate('Retrain', (0, 0), xytext=(7, -3), textcoords='offset points',
                fontsize=8, color=INK)
    ax.set_xlim(-.5, 6.2)
    ax.set_xlabel('$|\\Delta \\mathcal{D}_f|$ to Retrain (pts)  $\\rightarrow$ worse')
    ax.set_ylabel('test accuracy $-$ Retrain (pts)')
    ax.set_title('Classification: CIFAR-10, 10% instance', loc='left', color=INK)
    tidy(ax)

    ax = axes[1]
    goffsets = {'salun_port': (9, -11), 'random_mask': (9, -1), 'mmu_amp10': (9, 2),
                'mmu_2t_hi': (9, 9), 'mmu_nested': (9, 2)}
    for key, label in gnames.items():
        det = [gen[key][str(s)]['detected'] for s in SEEDS]
        clip = [gen[key][str(s)]['retain_clip'] for s in SEEDS]
        ours = key.startswith('mmu')
        ax.errorbar(np.mean(det), np.mean(clip),
                    xerr=np.std(det, ddof=1), yerr=np.std(clip, ddof=1),
                    fmt='o', color=AQUA if ours else MUTED,
                    markersize=8 if ours else 5.5, elinewidth=1, capsize=2,
                    markeredgecolor='white', markeredgewidth=1.2, zorder=4 if ours else 3)
        ax.annotate(label, (np.mean(det), np.mean(clip)), xytext=goffsets[key],
                    textcoords='offset points', fontsize=7.5,
                    color=INK if ours else MUTED)
    ax.set_xlabel('residual detections / 20  $\\rightarrow$ worse')
    ax.set_ylabel('retained CLIP cosine')
    ax.set_title('Generation: SD v1.4 concept erasure', loc='left', color=INK)
    ax.set_xlim(-.6, 6.4)
    tidy(ax)
    save(fig, 'fig5_cross_domain')


if __name__ == '__main__':
    DATA.mkdir(parents=True, exist_ok=True)
    fig_prefix_depth()
    fig_ulira_roc()
    fig_saliency_null()
    fig_margin_dial()
    fig_cross_domain()
