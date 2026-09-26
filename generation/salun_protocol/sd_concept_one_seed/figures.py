import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
ROWS = {'source': 'Source SD v1.4',
        'salun_port': 'SalUn',
        'random_mask': 'Random mask',
        'mmu_amp10': 'MMU'}
BLOCKS = [(0, 'fig_forget_1'), (10, 'fig_forget_2'),
          (20, 'fig_retain_1'), (30, 'fig_retain_2')]
MARK_DETECTIONS = True
SETS = {0: ('FORGET SET (removed): nude adult women, P1–P10', '#c62828'),
        10: ('FORGET SET (removed): nude adult men, P11–P20', '#c62828'),
        20: ('RETAIN SET (kept): clothed people, P21–P30', '#1a7f37'),
        30: ('RETAIN SET (kept): other objects, P31–P40', '#1a7f37')}


def main():
    data = json.loads((ROOT/'evaluation.json').read_text())
    hits = {n: {c['id'] for c in data[n]['records'] if c['detected']} for n in ROWS}
    for start, stem in BLOCKS:
        fig, axes = plt.subplots(len(ROWS), 10, figsize=(20, 2.06*len(ROWS)))
        fig.subplots_adjust(left=.088, right=.999, bottom=.002, top=.972,
                            wspace=.02, hspace=.035)
        for i, (name, label) in enumerate(ROWS.items()):
            for j in range(10):
                cid = start + j
                ax = axes[i, j]
                ax.imshow(Image.open(ROOT/'previews'/name/f'{cid:03}.png'))
                ax.set_xticks([]); ax.set_yticks([])
                mark = MARK_DETECTIONS and start < 20 and cid in hits[name]
                for sp in ax.spines.values():
                    sp.set_visible(mark)
                    if mark:
                        sp.set_color('red'); sp.set_linewidth(3)
                if i == 0:
                    ax.set_title(f'P{cid+1}', fontsize=13, color=SETS[start][1])
                    if j == 0:
                        ax.text(0, 1.3, SETS[start][0], transform=ax.transAxes,
                                fontsize=15, fontweight='bold', color=SETS[start][1])
                if j == 0:
                    ax.set_ylabel(label, fontsize=13)
        for ext in ('pdf', 'png'):
            fig.savefig(ROOT/f'{stem}.{ext}', dpi=150, bbox_inches='tight',
                        facecolor='white')
        plt.close(fig)
        print('wrote', stem)
    combined(data, hits)


def combined(data, hits):
    n = len(ROWS)
    fig = plt.figure(figsize=(20, 4.12*n))
    blocks = [fig.add_gridspec(n, 10, left=.088, right=.999, top=t, bottom=b,
                               wspace=.02, hspace=.035)
              for t, b in [(.982, .525), (.468, .011)]]
    for bi, (gs, start) in enumerate(zip(blocks, [0, 10])):
        for i, (name, label) in enumerate(ROWS.items()):
            for j in range(10):
                cid = start + j
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(Image.open(ROOT/'previews'/name/f'{cid:03}.png'))
                ax.set_xticks([]); ax.set_yticks([])
                mark = MARK_DETECTIONS and cid in hits[name]
                for sp in ax.spines.values():
                    sp.set_visible(mark)
                    if mark:
                        sp.set_color('red'); sp.set_linewidth(3)
                if i == 0:
                    ax.set_title(f'P{cid+1}', fontsize=13, color=SETS[start][1])
                    if j == 0:
                        ax.text(0, 1.3, SETS[start][0], transform=ax.transAxes,
                                fontsize=15, fontweight='bold', color=SETS[start][1])
                if j == 0:
                    ax.set_ylabel(label, fontsize=13)
    for ext in ('pdf', 'png'):
        fig.savefig(ROOT/f'fig_forget_all.{ext}', dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('wrote fig_forget_all')


if __name__ == '__main__':
    main()
