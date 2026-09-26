import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
out=root/'results/improvement_20260907'
plan={'candidate_tag':'validate_b4', 'candidate_beta':4, 'candidate_warmup':0,
      'baseline_seed42':str(root/'results/full_redirect_20260907_cfg2_mmu_fullwidth_c0_seed42'),
      'candidate_seed42':str(root/'results/improve_b4_c0_seed42'),
      'additional_seeds':[43,44], 'guidance':2,
      'rationale':'Beta=4 improves seed-42 UA and FID versus beta=1, with retained accuracy lower by 0.42 percentage points. Saved images remain recognizable; no noise-only collapse. Warm-up helps nesting but does not beat the equally trained full-width control.'}
(out/'validation_plan.json').write_text(json.dumps(plan,indent=2))
for seed in [43,44]:
 for tag,beta in [('validate_base',1),('validate_b4',4)]:
  subprocess.run([sys.executable,'-u',str(root/'run_experiment.py'),'--methods','mmu_fullwidth',
                  '--beta',str(beta),'--guidance','2','--seed',str(seed),'--tag',tag],
                 cwd=root.parent,check=True)
  run=root/f'results/{tag}_c0_seed{seed}'
  subprocess.run([sys.executable,str(root/'summarize_run.py'),str(run)],check=True)
subprocess.run([sys.executable,str(root/'summarize_improvements.py')],check=True)
print('VALIDATION COMPLETE',flush=True)
