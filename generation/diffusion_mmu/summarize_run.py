import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

LABELS = {'source':'Source', 'retrain':'Retrain reference', 'finetune':'Fine-tune',
          'neggrad':'NegGrad', 'salun':'SalUn (local adaptation)',
          'mmu_fullwidth':'MMU full width', 'mmu':'MMU nested',
          'mmu_noskip':'MMU untruncated skips', 'mmu_retainonly':'MMU retain only'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    a = ap.parse_args()
    results = json.loads((a.run/'results.json').read_text())
    config = json.loads((a.run/'config.json').read_text())
    grids = torch.load(a.run/'grids.pt', map_location='cpu', weights_only=True)
    lines = [f"# Diffusion evaluation: guidance {config['guidance']}", '',
             f"Forget class {config['forget_class']}; seed {config['seed']}; "
             f"{config['n_ua']} forgotten-condition samples; {config['n_fid']} retained samples.", '',
             '| Method | UA (%) | Retained FID | Retained accuracy (%) | Train/load time (s) |',
             '|---|---:|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f"| {LABELS[name]} | {r['UA']:.1f} | {r['FID_retain']:.2f} | "
                     f"{r['retain_cond_acc']:.1f} | {r['unlearn_time_s']:.1f} |")
    lines += ['', 'UA alone cannot distinguish valid redirection from damaged generation.',
              'SalUn is a local adaptation, not a validated reproduction. One seed and one class do not establish general superiority.']
    (a.run/'summary.md').write_text('\n'.join(lines)+'\n')
    for kind in ('forget', 'retain'):
        fig, axes = plt.subplots(len(results), 8, figsize=(11, 1.25*len(results)), squeeze=False)
        for row, name in enumerate(results):
            imgs = grids[name][kind]
            for col in range(8):
                ax = axes[row,col]
                ax.imshow(((imgs[col].permute(1,2,0)+1)/2).clamp(0,1))
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(LABELS[name], fontsize=9)
        fig.suptitle(f'{kind.capitalize()} conditions — guidance {config["guidance"]}')
        fig.tight_layout()
        fig.savefig(a.run/f'samples_{kind}.png', dpi=150, bbox_inches='tight')
        fig.savefig(a.run/f'samples_{kind}.pdf', bbox_inches='tight')
        plt.close(fig)
    print((a.run/'summary.md').read_text())

if __name__ == '__main__':
    main()
