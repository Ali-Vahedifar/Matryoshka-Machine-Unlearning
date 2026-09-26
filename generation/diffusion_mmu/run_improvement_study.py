import json
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = root/'results/improvement_20260907'
manifest.mkdir(exist_ok=True)
code = manifest/'code'
code.mkdir(exist_ok=True)
for name in ['unlearn.py','run_experiment.py','ddpm.py','evaluate.py','test_revision.py',Path(__file__).name]:
    shutil.copy2(root/name, code/name)
plan = {'seed':42, 'forget_class':0, 'guidance':2, 'epochs':3,
        'n_ua':500, 'n_fid':5000, 'steps':100,
        'beta_sweep':[0.5,1,2,4], 'warmup_epochs':2,
        'beta1_reference':str(root/'results/full_redirect_20260907_cfg2_mmu_fullwidth_c0_seed42'),
        'selection':'Inspect the retained-quality/UA tradeoff and forgotten-condition grids; validate selected configurations on seeds 43 and 44 with paired beta=1 controls. No claim of independent source-model seeds.'}
(manifest/'plan.json').write_text(json.dumps(plan,indent=2))
jobs = [('improve_b05',['mmu_fullwidth'],.5,0),
        ('improve_b2',['mmu_fullwidth'],2,0),
        ('improve_b4',['mmu_fullwidth'],4,0),
        ('improve_warm2',['mmu_fullwidth','mmu'],1,2)]
for tag,methods,beta,warmup in jobs:
    args=[sys.executable,'-u',str(root/'run_experiment.py'),'--methods',','.join(methods),
          '--beta',str(beta),'--warmup_epochs',str(warmup),'--guidance','2',
          '--seed','42','--tag',tag]
    print('RUN',args,flush=True)
    subprocess.run(args,cwd=root.parent,check=True)
    run=root/f'results/{tag}_c0_seed42'
    subprocess.run([sys.executable,str(root/'summarize_run.py'),str(run)],check=True)
print('STAGE ONE COMPLETE',flush=True)
