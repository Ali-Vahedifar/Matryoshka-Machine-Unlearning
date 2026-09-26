import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

root = Path(__file__).resolve().parent
out = root/'results/improvement_20260907'
out.mkdir(exist_ok=True)
specs = [
 ('Full width beta=0.5', 'improve_b05', 'mmu_fullwidth'),
 ('Full width beta=1', 'full_redirect_20260907_cfg2_mmu_fullwidth', 'mmu_fullwidth'),
 ('Full width beta=2', 'improve_b2', 'mmu_fullwidth'),
 ('Full width beta=4', 'improve_b4', 'mmu_fullwidth'),
 ('Full width warm-up=2', 'improve_warm2', 'mmu_fullwidth'),
 ('Nested warm-up=0', 'full_redirect_20260907_cfg2_mmu', 'mmu'),
 ('Nested warm-up=2', 'improve_warm2', 'mmu'),
]
rows=[]; grids=[]
for label,tag,name in specs:
 run=root/f'results/{tag}_c0_seed42'
 if not (run/'results.json').exists(): continue
 data=json.loads((run/'results.json').read_text())
 if name not in data: continue
 r=data[name]
 rows.append({'label':label,'run':str(run),'method':name,**r})
 grids.append((label,torch.load(run/'grids.pt',map_location='cpu',weights_only=True)[name]))
(out/'seed42_trials.json').write_text(json.dumps(rows,indent=2))
lines=['# MMU improvement study','',
       'CIFAR-10 airplane forgetting; guidance 2; 100 DDIM steps; 500 forgotten-condition and 5,000 retained samples per evaluation. '
       'Three joint epochs; warm-up adds two retained-only epochs with the same optimizer. '
       'The beta=1 and nested no-warm-up references reuse the completed experiment.','',
       '| Seed-42 configuration | UA (%) ↑ | Retained FID ↓ | Retained accuracy (%) ↑ |',
       '|---|---:|---:|---:|']
for r in rows:
 lines.append(f"| {r['label']} | {r['UA']:.1f} | {r['FID_retain']:.2f} | {r['retain_cond_acc']:.2f} |")
lines += ['', 'Seed 42 is the tuning run. Additional seeds share the same pretrained source checkpoint; '
          'they measure variation in retained subset selection, unlearning and sampling, not source training. '
          'Higher UA alone does not establish improved unlearning.','']
for kind in ['forget','retain']:
 if not grids: continue
 fig,axes=plt.subplots(len(grids),8,figsize=(11,len(grids)*1.2),squeeze=False)
 for i,(label,g) in enumerate(grids):
  for j in range(8):
   ax=axes[i,j]; ax.imshow(((g[kind][j].permute(1,2,0)+1)/2).clamp(0,1))
   ax.set_xticks([]); ax.set_yticks([])
   if j==0: ax.set_ylabel(label,fontsize=9)
 fig.suptitle(f'{kind.capitalize()} conditions — improvement trials, seed 42, guidance 2')
 fig.tight_layout()
 for ext in ['png','pdf']: fig.savefig(out/f'samples_{kind}.{ext}',dpi=150,bbox_inches='tight')
 plt.close(fig)
lines += ['[Forgotten-condition grid](samples_forget.png)','', '[Retained-condition grid](samples_retain.png)','']
validation=out/'validation_plan.json'
if validation.exists():
 plan=json.loads(validation.read_text()); candidate=plan['candidate_tag']
 groups={}
 for label,tag in [('Baseline beta=1','validate_base'),('Selected candidate',candidate)]:
  group=[]
  seed42=Path(plan['baseline_seed42'] if label.startswith('Baseline') else plan['candidate_seed42'])
  for seed,run in [(42,seed42)]+[(seed,root/f'results/{tag}_c0_seed{seed}') for seed in [43,44]]:
   if (run/'results.json').exists():
    r=json.loads((run/'results.json').read_text())['mmu_fullwidth']; group.append((seed,r))
  groups[label]=group
 lines += ['## Paired-seed comparison','', '| Configuration | Seed | UA (%) ↑ | Retained FID ↓ | Retained accuracy (%) ↑ |', '|---|---:|---:|---:|---:|']
 for label,group in groups.items():
  for seed,r in group:
   lines.append(f"| {label} | {seed} | {r['UA']:.1f} | {r['FID_retain']:.2f} | {r['retain_cond_acc']:.2f} |")
  if len(group)==3:
   vals=[]
   for metric in ['UA','FID_retain','retain_cond_acc']:
    v=[r[metric] for _,r in group]
    vals.append(f'{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}')
   lines.append(f"| {label} | mean ± sample SD | "+' | '.join(vals)+' |')
 lines += ['', 'The three-seed summary includes the tuning seed. The seed-43/44 rows are the additional checks.', '']
(out/'summary.md').write_text('\n'.join(lines))
print('\n'.join(lines))
