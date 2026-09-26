import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(args,label):
    log=ROOT/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
    print('RUN',label,flush=True)
    with log.open('a') as f:
        subprocess.run([sys.executable,'-u',str(ROOT/'classification.py'),*args],
                       stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT.parent)

def main():
    for ratio in ['.1','.5']:
        run(['--seed','1','--ratio',ratio,'--tune','--methods','retrain,MMU2T'],f'mmu2t_tune_{ratio}')
        sel=ROOT/f'classification/seed1/forget{int(float(ratio)*100)}/tuning/selected.json'
        print(ratio,'selected',json.loads(sel.read_text()).get('MMU2T'),flush=True)
    for seed in range(2,12):
        for ratio in ['.1','.5']:
            sel=ROOT/f'classification/seed1/forget{int(float(ratio)*100)}/tuning/selected.json'
            run(['--seed',str(seed),'--ratio',ratio,'--settings',str(sel),'--methods','MMU2T'],
                f'mmu2t_seed{seed}_{ratio}')
    print('MMU2T STUDY COMPLETE',flush=True)

if __name__=='__main__':main()
