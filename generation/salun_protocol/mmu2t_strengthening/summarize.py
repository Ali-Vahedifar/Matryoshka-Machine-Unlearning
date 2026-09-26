import os
import json,logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
logging.getLogger('fontTools').setLevel(logging.WARNING)
R=Path(__file__).resolve().parent;W=Path(os.environ.get('MMU_WORK','work')+'/mmu2t_strengthening/generation')
NAMES=['MMU','Attraction only','Nested MMU','Nested attraction','Random mask','SalUn'];COLORS=['#1765D1','#9670B7','#5BA0DE','#CA9ED4','#139888','#E66A33']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def main():
 g=json.loads((R/'generation_results.json').read_text());cl=json.loads((R/'classification_results.json').read_text());sd=json.loads((R/'sd_heldout/results.json').read_text());figs=[]
 lines=['# Two-teacher MMU: strengthening study','', 'Reference: latest full-width two-teacher MMU, fixed margin 0.002. Generation training seed 42; classification seed 2 reuses existing paired source/retrain models. No seed sweep or test-based hyperparameter selection. These are exploratory results, including negative outcomes.','',
 '## 1. Component ablation','', 'Three fixed deletion tasks: airplane, automobile, bird. Full-width versus nested student prefixes [0.5, 0.75, 1.0], crossed with attraction only versus attraction plus repulsion. Nested variants share full-width frozen-source targets; they are ablations, not the reference MMU. Baselines use the existing local masked objective. Each run has 1,000 updates and 128 retained + 128 forgotten examples per full batch; partial batches are preserved. Nested gradients average across widths.','', '| Method | D_f condition accuracy (%) ↓ | D_r condition accuracy (%) ↑ | Mean training seconds ↓ |','|---|---:|---:|---:|']
 endpoint={}
 for name in NAMES:
  rs=[g[f'ablation/class{c}/{name.replace(" ","_")}/step1000'] for c in range(3)]
  f=100*sum(r['per_class'][str(c)]['correct'] for c,r in enumerate(rs))/48;ret=100*sum(r['retain_correct'] for r in rs)/432
  times=[json.loads((W/f'ablation/class{c}'/name.replace(' ','_')/'history.json').read_text())[-1]['seconds'] for c in range(3)]
  endpoint[name]=[f,ret,float(np.mean(times))];lines.append(f'| {name} | {f:.2f} | {ret:.2f} | {np.mean(times):.1f} |')
 lines+=['','The CIFAR masked baselines were rerun in this suite’s training loop. Their stochastic draw order differs from the earlier ten-class driver, despite the same numeric seed and loss settings. Treat these as new trajectories, not exact reproductions of the earlier baseline checkpoints; do not pool the two tables as independent seeds.', '', 'Training time includes checkpoint writes, excludes evaluation and mask construction. Baseline saliency computation is additional overhead. Matched update counts are not equal FLOPs; nested variants use three student widths. These are local baseline implementations.','', '## 2. Update-budget curves','', 'All checkpoints at 100, 250, 500, and 1,000 updates are reported; no best checkpoint is selected using these evaluation samples. Curves average counts over three deletion tasks (48 forgotten and 432 retained generated images per checkpoint). No statistical superiority is inferred from task averages.','']
 fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for name,col in zip(NAMES,COLORS):
  f=[];r=[]
  for s in [100,250,500,1000]:
   rs=[g[f'ablation/class{c}/{name.replace(" ","_")}/step{s}'] for c in range(3)]
   f.append(100*sum(v['per_class'][str(c)]['correct'] for c,v in enumerate(rs))/48);r.append(100*sum(v['retain_correct'] for v in rs)/432)
  axs[0].plot([100,250,500,1000],f,marker='o',color=col,label=name);axs[1].plot(f,r,marker='o',color=col,label=name)
 axs[0].set(xlabel='Optimizer updates',ylabel='Forgotten-condition accuracy (%) ↓',title='Forgetting across update budgets')
 axs[1].set(xlabel='Forgotten-condition accuracy (%) ↓',ylabel='Retained-condition accuracy (%) ↑',title='Forgetting–retention trajectories')
 for ax in axs:ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
 fig.savefig(R/'budget_curves.pdf',bbox_inches='tight');fig.savefig(R/'budget_curves.png',dpi=180,bbox_inches='tight');figs.append(fig)
 lines+=['[Budget and trade-off curves](budget_curves.pdf)','', '## 3. Sequential versus joint deletion','', 'Sequential order is airplane → automobile → bird, 1,000 updates per stage. Each stage receives only the newly forgotten class; retained batches exclude all classes removed so far. There is no forgotten-data rehearsal. All deleted conditions redirect toward truck, which remains retained. The frozen original source supplies MMU targets throughout. Joint deletion uses all three forgotten classes together for 3,000 updates. Both have the same total optimizer budget, but data exposure necessarily differs. A single fixed order is tested.','', '| Method | Route | Airplane still predicted /16 ↓ | Automobile /16 ↓ | Bird /16 ↓ | D_r accuracy (%) ↑ |','|---|---|---:|---:|---:|---:|']
 fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True,layout='constrained')
 for name,col,ax in zip(['MMU','Random mask','SalUn'],[COLORS[0],COLORS[4],COLORS[5]],axs):
  key=name.replace(' ','_')
  for route in ['sequential/stage3','joint']:
   v=g[f'{route}/{key}'];values=[v['per_class'][str(c)]['correct'] for c in range(3)]
   lines.append(f'| {name} | {route} | {values[0]} | {values[1]} | {values[2]} | {100*v["retain_correct"]/v["retain_n"]:.2f} |')
  for c,label in enumerate(['Airplane','Automobile','Bird']):
   stages=list(range(c+1,4));vals=[100*g[f'sequential/stage{s}/{key}']['per_class'][str(c)]['correct']/16 for s in stages]
   ax.plot(stages,vals,marker=['o','s','^'][c],label=label)
  ax.set_title(name,color=col);ax.set_xticks([1,2,3],['Remove air','+ auto','+ bird']);ax.set_ylim(-3,103);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
 axs[0].set_ylabel('Forgotten class still predicted (%) ↓');fig.savefig(R/'sequential_reappearance.pdf',bbox_inches='tight');figs.append(fig)
 lines+=['','[Reappearance by deletion stage](sequential_reappearance.pdf)','', '## 4. Held-out generation','', 'CIFAR: sampling seeds 60042 and 60043, fixed before evaluation; same three deletion tasks and full-width final models. No image selection. Each method has 96 forgotten and 864 retained outputs over tasks and draws.','', '| Method | D_f accuracy (%) ↓ | D_r accuracy (%) ↑ | Forgotten-class predictions under retained conditions /864 ↓ |','|---|---:|---:|---:|']
 for name in ['MMU','Random mask','SalUn']:
  rows=[(c,g[f'heldout/class{c}/{name.replace(" ","_")}/{seed}']) for c in range(3) for seed in [60042,60043]]
  fc=sum(v['per_class'][str(c)]['correct'] for c,v in rows);rc=sum(v['retain_correct'] for c,v in rows)
  leakage=sum(sum(pred==c for k,row in v['per_class'].items() if int(k)!=c for pred in row['predictions']) for c,v in rows)
  lines.append(f'| {name} | {100*fc/96:.2f} | {100*rc/864:.2f} | {leakage} |')
 lines+=['','Stable Diffusion: 20 new nonsexual adult-art prompts (paraphrases/compositions), plus 20 new retained prompts, fixed seed-42 checkpoints. Source, MMU, Random mask, and SalUn share sampling noise. Detector thresholds remain 0.6 for scores and 0.2 for censored previews. CLIP is alignment, not comprehensive quality.','', '| Method | Forgotten detections /20 ↓ | Retained CLIP alignment ↑ |','|---|---:|---:|']
 for name,v in sd.items():lines.append(f'| {name} | {v["detected"]} | {v["retain_clip"]:.4f} |')
 lines+=['','[Held-out SD details and censored images](sd_heldout/report.md)','', '## 5. Retrain-calibrated classification','', 'This is an explicit classification adaptation: a retain-only fine-tuned source supplies attraction targets on forgotten examples; the original frozen source supplies repulsion and retained preservation targets. The good teacher is trained for five epochs without forgotten data, but is initialized from a source that saw it. It is not the retrain oracle. MMU and attraction-only use five joint epochs, lr 0.001, KL temperature 1, margin 0.002 for MMU. Teacher construction cost is included separately. The full-width SCRUB-style control uses the prior local max/min recipe and temperature 4. Protocols are not interchangeable.','', 'D_f = forgotten-set accuracy; D_r = retained-set accuracy. Average gap = mean absolute percentage-point deviation from paired retrain over D_f, D_r, Test accuracy and released confidence-SVC MIA efficacy. For random forgetting, closer to retrain is preferred; D_f should not be driven blindly to zero.','', '| Forget ratio | Method | D_f | D_r | Test accuracy | MIA efficacy | Avg. gap ↓ | Training minutes ↓ |','|---|---|---:|---:|---:|---:|---:|---:|']
 for ratio,rows in cl.items():
  for name,v in rows.items():
   minutes='—' if name=='Source' or v['seconds'] is None else f"{(v['seconds']+v['teacher_seconds'])/60:.2f}"
   lines.append(f'| {ratio}% | {name} | {v["Df_accuracy"]:.2f} | {v["Dr_accuracy"]:.2f} | {v["Test_accuracy"]:.2f} | {v["MIA"]:.2f} | {v["avg_gap"]:.2f} | {minutes} |')
 lines+=['','## Scope and reproducibility','', 'Classification times include good-teacher construction for the two-teacher adaptation and attraction-only control; reused SalUn and retrain timings are historical. Sequential masks use the fixed original source on each new forgotten subset. Nested history `loss` records the mean scaled width contribution; its full averaged objective is the sum of recorded attraction, hinge, and retain terms (logging convention only). No additional training seeds, test-set tuning, or automatic winner selection. Sampling seeds are not independent training trials. All methods, checkpoints, and failure cases remain reported. Generation uses a fixed local CIFAR DDPM and local SD implementations. This study does not establish certified removal, comprehensive perceptual quality, or broad superiority.','', 'Scripts and fixed protocols are beside this report. Raw checkpoints, sampled tensors, and per-sample predictions are in `$MMU_WORK/mmu2t_strengthening`. All original experiment results remain available.']
 (R/'report.md').write_text('\n'.join(lines)+'\n')
 with PdfPages(R/'analysis.pdf') as pdf:
  for fig in figs:pdf.savefig(fig,bbox_inches='tight')
 (R/'status.json').write_text(json.dumps(dict(status='complete',groups=['ablation','budget_curves','sequential_joint','heldout_generation','classification'])))
 print('REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
