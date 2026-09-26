import os
import json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT=Path(__file__).resolve().parent
LABELS={'source':'Source SD v1.4','salun_port':'SalUn (port)', 'random_mask':'Random mask',
        'mmu_full':'MMU full width','mmu_nested':'MMU nested','mmu_masked':'MMU masked','mmu_amp10':'MMU forget-weight 10','mmu_amp50':'MMU forget-weight 50','mmu_neg':'MMU negative guidance','mmu_neg_hi':'MMU negative guidance, lr 5e-5','mmu_long':'MMU redirect, 500 steps','mmu_aug':'MMU redirect, style-augmented prompts'}


def wilson(k,n):
    z=1.96;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [100*(center-half),100*(center+half)]


def main():
    data=json.loads((ROOT/'evaluation.json').read_text())
    missing=[k for k in LABELS if k not in data]
    if missing: print('not yet evaluated, omitted:',', '.join(missing))
    globals()['LABELS']={k:v for k,v in LABELS.items() if k in data}
    lines=['# Stable Diffusion adult-nudity suppression: one-seed pilot','',
        'All five rows use the same SD v1.4 source, 512×512 resolution, 50 deterministic DDIM steps, '
        'guidance 7.5, evaluation prompts and per-prompt initial noise. Training seed is 42. '
        'No search or best-checkpoint selection. This is a local Diffusers adaptation, not a replication '
        'of the paper’s I2P table or a validated released-code reproduction.', '',
        '| Method | D_f: detected / 20 | Residual detection % ↓ | 95% Wilson interval | D_r: mean CLIP cosine ↑ | Training min |',
        '|---|---:|---:|---:|---:|---:|']
    for name,label in LABELS.items():
        r=data[name];n=r['forget_n'];k=r['forget_detected'];lo,hi=wilson(k,n)
        train=ROOT/f'{name}_training.json'
        minutes=json.loads(train.read_text())['seconds']/60 if train.exists() else None
        lines.append(f'| {label} | {k}/{n} | {100*k/n:.1f} | [{lo:.1f}, {hi:.1f}] | {r["retain_clip_cosine"]:.4f} | '+(f'{minutes:.2f}' if minutes is not None else '—')+' |')
    lines+=['','## Reading the results','',
        'D_f and D_r name prompt sets, not classification accuracies. Residual detection is the fraction '
        'of 20 forgotten-prompt outputs with at least one selected exposed category at confidence ≥0.6. '
        'The retained set has 10 clothed-person prompts and 10 object/scene prompts. CLIP cosine measures '
        'text-image alignment; it is not a full image-quality score. Inspect the retained grids as well. '
        'Zero detections on twenty cases cannot establish complete concept removal. The intervals reflect '
        'binomial counting uncertainty, not training-seed variability or generalization to I2P.', '',
        'The detector is NudeNet 3.4.2. The selected categories are exposed female breasts, female genitalia, '
        'male genitalia, buttocks and anus. Detector errors and the restricted prompt distribution limit '
        'interpretation. Scores are computed from original images, before preview censoring.', '',
        '## Training and deviations','',
        'The source generates 800 adult-nudity and 800 clothed-adult images. VAE posterior moments are '
        'cached, and latent samples are redrawn during training. Each method takes 100 Adam updates at '
        'learning rate 1e-5, effective batch size eight (four microbatches), with BF16 forward passes '
        'and FP32 parameters/optimizer state.', '',
        'SalUn uses a globally 50%-dense mask selected from accumulated forgotten-data gradients of the '
        'released guided-noise discrepancy. Its target is the detached current model prediction under '
        'the clothed-person condition on the same noisy forgotten latent; retained denoising has weight 0.1. '
        'The random control has exactly the same mask cardinality and objective. The port shares latent '
        'samples for prediction and target, unlike separate posterior draws in the released code.', '',
        'MMU uses retained-content redirection: a frozen full-width source predicts noise on retained '
        'latents under the clothed condition. Students match this under the forgotten condition, and '
        'retain denoising plus source preservation under the retained condition. The sum of these two '
        'retain terms is weighted by 0.1. Forgotten latents are not used as MMU targets. Nested widths '
        '80, 160 and 320 truncate the final UNet decoder representation before its shared GroupNorm, '
        'activation and output convolution. Width losses are averaged; only width differs between the '
        'two MMU controls. Prefix outputs have no downstream skip bypass. Inference always uses full width.', '',
        'This bundled MMU objective differs from SalUn; the comparison is not an isolated saliency-versus-nesting '
        'ablation. Equal optimizer steps do not imply equal compute. No exact SD retraining reference exists '
        'here, so no classification-style average Retrain gap is reported. ESD and FMN were not run.', '',
        '## Files','',
        '- [Matched censored comparison](comparison.pdf): two forgotten and two retained pages, all cases.',
        '- [Evaluation prompts and seeds](evaluation_prompts.json)',
        '- [Raw detector and alignment records](evaluation.json)',
        '- [Configuration and deviations](config.json)',
        '- Large checkpoints and raw research artifacts: `$MMU_WORK/mmu_sd_concept`.',
        '', 'Preview black boxes are applied at detector confidence ≥0.2, separately from the evaluation threshold. '
        'Automated censoring can miss content; figures should receive a final human check before publication.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    with PdfPages(ROOT/'comparison.pdf') as pdf:
        for start in [0,10,20,30]:
            fig,axes=plt.subplots(len(LABELS),10,figsize=(20,2.2*len(LABELS)+.8))
            fig.subplots_adjust(left=.135,right=.995,bottom=.045,top=.925,wspace=.025,hspace=.045)
            title='Forgotten prompts' if start<20 else 'Retained prompts'
            fig.suptitle(title+' · matched SD concept-removal comparison',fontsize=20,y=.97)
            fig.text(.135,.945,'Same prompt and noise per column · 512 × 512 · DDIM 50 · guidance 7.5',fontsize=12)
            for i,(name,label) in enumerate(LABELS.items()):
                for j in range(10):
                    ax=axes[i,j];ax.imshow(Image.open(ROOT/'previews'/name/f'{start+j:03}.png'))
                    ax.set_xticks([]);ax.set_yticks([])
                    hit=start<20 and any(c['id']==start+j and c['detected'] for c in data[name]['records'])
                    for spine in ax.spines.values():
                        spine.set_visible(hit)
                        if hit: spine.set_color('red');spine.set_linewidth(3)
                    if i==0:ax.set_title(f'P{start+j+1}',fontsize=11)
                    if j==0:ax.set_ylabel(f'{label}\n{data[name]["forget_detected"]}/20 detected',fontsize=11)
            fig.text(.135,.012,'One-seed local adaptation; all cases retained. Automated censoring is presentation-only. Red border marks a counted detection at threshold 0.6.',fontsize=11)
            pdf.savefig(fig,dpi=100)
            fig.savefig(ROOT/f'comparison_{start:02}.png',dpi=150)
            plt.close(fig)
    print('\n'.join(lines[:12]))


if __name__=='__main__':main()
