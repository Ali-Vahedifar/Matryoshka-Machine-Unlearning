import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def status(**kw):
    p=ROOT/'classification_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'updated_utc':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),**kw},indent=2));tmp.replace(p)


def run(args,label):
    log=ROOT/'logs'/f'{label}.log'
    status(state='running',job=label,log=str(log))
    with log.open('a') as f:
        subprocess.run([sys.executable,'-u',str(ROOT/'classification.py'),*args],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT.parent)


def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    if not (ROOT/'classification/seed1/source.pt').exists():
        run(['--seed','1','--ratio','.1','--source_only'],'source_seed1')
    for ratio in ['.1','.5']:
        selected=ROOT/f'classification/seed1/forget{int(float(ratio)*100)}/tuning/selected.json'
        if not selected.exists():run(['--seed','1','--ratio',ratio,'--tune'],f'tune_{ratio}')
    for seed in range(2,12):
        for ratio in ['.1','.5']:
            settings=ROOT/f'classification/seed1/forget{int(float(ratio)*100)}/tuning/selected.json'
            run(['--seed',str(seed),'--ratio',ratio,'--settings',str(settings)],f'trial_seed{seed}_{ratio}')
            subprocess.run([sys.executable,str(ROOT/'summarize_classification.py')],check=True)
    status(state='complete',job='all_10_trials_both_ratios')

if __name__=='__main__':
    try:main()
    except Exception as e:
        current=json.loads((ROOT/'classification_status.json').read_text())
        status(state='failed',job=current.get('job'),log=current.get('log'),error=repr(e))
        raise
