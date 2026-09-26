import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
while not (R/'generation_status.json').exists():
 try:os.kill(2180787,0)
 except ProcessLookupError:raise RuntimeError('Generation exited without completion; inspect generation.log')
 time.sleep(30)
for name,python in [('classification_study',sys.executable),('sd_heldout',os.environ.get('MMU_WORK','work')+'/mmu_sd_concept/venv/bin/python')]:
 env=dict(os.environ,HF_HOME=os.environ.get('MMU_WORK','work')+'/mmu_sd_concept/hf',HF_HUB_DISABLE_XET='1')
 with (R/f'{name}.log').open('a') as f:subprocess.run([python,'-u',str(R/f'{name}.py')],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
subprocess.run([sys.executable,str(R/'summarize.py')],check=True)
print('ALL STUDIES COMPLETE',flush=True)
