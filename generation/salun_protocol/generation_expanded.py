import json
import textwrap
import time
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from generation_figure import OUT as BASE, D, U, Diffusion, generate, CLASSES

OUT = BASE/'expanded'


def panel(images, cls, start, count=8):
    names = list(images)
    fig, axes = plt.subplots(len(names), count, figsize=(13, 10.5))
    fig.subplots_adjust(left=.23, right=.99, bottom=.045, top=.91, wspace=.06, hspace=.13)
    status = 'Forgotten condition' if cls == 0 else 'Retained condition'
    fig.suptitle(f'{status}: {CLASSES[cls]}', x=.23, ha='left', fontsize=20, fontweight='bold')
    fig.text(.23, .935, 'Matched noise across methods · 1,000 DDIM steps · guidance 2', fontsize=10, color='#444444')
    for i, (name, data) in enumerate(images.items()):
        for j in range(count):
            ax = axes[i, j]
            ax.imshow(((data[cls, start+j].permute(1,2,0)+1)/2).clamp(0,1), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(f'Sample {start+j+1:02}', fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(textwrap.fill(name, 10), labelpad=14, fontsize=12, fontweight='bold')
    fig.text(.23, .014, 'Native 32 × 32 outputs; no enhancement or sample selection. Local DDPM comparison.', fontsize=9, color='#555555')
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = {
        'Source': D/'ckpt/source.pt',
        'Retrain reference': D/'ckpt/retrain_c0.pt',
        'SalUn (local)': BASE/'salun.pt',
        'Random mask (local)': BASE/'random.pt',
        'MMU full width, beta=1': D/'results/full_redirect_20260907_c0_seed42/mmu_fullwidth.pt',
        'MMU full width, beta=4': D/'results/improve_b4_c0_seed42/mmu_fullwidth.pt',
        'MMU nested, warm-up=2': D/'results/improve_warm2_c0_seed42/mmu.pt'}
    config = dict(seed=10043, previous_seed=10042, new_samples_per_class=16,
                  total_samples_per_class=32, steps=1000, guidance=2, batch_size=160,
                  checkpoints={k:str(v) for k,v in checkpoint.items()},
                  note='Same sampler as original figure. No training, filtering, enhancement or cherry-picking.')
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    old = torch.load(BASE/'samples.pt', map_location='cpu', weights_only=False)
    labels = torch.arange(10).repeat_interleave(16)
    assert torch.equal(old['labels'], labels)
    fresh_path = OUT/'new_samples.pt'
    fresh = torch.load(fresh_path, map_location='cpu', weights_only=False)['images'] if fresh_path.exists() else {}
    for name, path in checkpoint.items():
        if name in fresh:
            continue
        start = time.time()
        model = U.load_source(path)
        torch.manual_seed(config['seed'])
        fresh[name] = generate(model, Diffusion(), labels, bs=160, guidance=2, steps=1000).cpu()
        assert fresh[name].shape == (160,3,32,32) and torch.isfinite(fresh[name]).all()
        torch.save(dict(labels=labels, images=fresh, config=config), fresh_path)
        del model
        print(name, 'generated 160 additional samples in', round(time.time()-start,1), 'seconds', flush=True)
    images = {name:torch.cat([old['images'][name].cpu().reshape(10,16,3,32,32),
                              fresh[name].reshape(10,16,3,32,32)], dim=1) for name in checkpoint}
    with PdfPages(OUT/'all_samples.pdf') as atlas, PdfPages(OUT/'comparison.pdf') as main_pdf:
        for cls in range(10):
            for start in [0,8,16,24]:
                fig = panel(images, cls, start)
                atlas.savefig(fig, dpi=300)
                if start == 16:
                    main_pdf.savefig(fig, dpi=300)
                    if cls in [0,1]:
                        fig.savefig(OUT/f'{CLASSES[cls]}_comparison.png', dpi=300)
                plt.close(fig)
            print('Rendered', CLASSES[cls], flush=True)
    (OUT/'README.md').write_text('''# Expanded generation comparison

Seven methods, ten conditions, 32 samples per condition per method: **2,240 images**.
1,120 are new draws (seed 10043); 1,120 are preserved prior draws (seed 10042).
The comparison uses the same checkpoints, 1,000 DDIM steps and guidance 2.

- [Comparison: ten larger pages](comparison.pdf): first eight new draws for every class.
- [Complete atlas: forty pages](all_samples.pdf): every old and new draw, in order.
- [Airplane comparison](airplane_comparison.png)
- [Automobile comparison](automobile_comparison.png)

Every column uses matched noise across methods. Native image size is 32 × 32;
nearest-neighbor rendering preserves those pixels, and PDF labels remain vector text.
These exports improve presentation, not the trained model's generation quality.
No draws were discarded. The nested-model degradation and retrain-reference
behavior under the forgotten condition remain visible. These are local DDPM
adaptations, with the previously documented unequal training budgets; no new
FID evaluation or superiority claim is implied.
''')
    print('EXPANDED GENERATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
