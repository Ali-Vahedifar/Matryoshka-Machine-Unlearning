import os
import copy,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'diffusion_mmu'))
import unlearn as U
from ddpm import Diffusion
from train_source import cifar10
from evaluate import generate,judge,classify,CLASSES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
OUT=ROOT/'two_teacher_seed42';WORK=Path(os.environ.get('MMU_WORK','work')+'/mmu_cifar_two_teacher')

def cycle(x,y):
    while True: yield from U.batches(x,y,128)

def main():
    OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True);torch.set_num_threads(4)
    config=dict(seed=42,steps=1000,lr=1e-4,batch_size=128,margin=.002,
                retain_weight=3.072,preservation_weight=3.072,gradient_clip=1.,
                widths=[1.],good_condition=1,bad_condition=0,sampling_steps=1000,
                guidance=2,sampling_seeds=[10042,10043],samples_per_class=32,
                note='Full width only; frozen source supplies both roles. Retain denoising coefficient matches CIFAR SalUn sum-pixel loss (0.001*3072); equal additional teacher-preservation coefficient. Fixed transferred margin, not tuned.')
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    source=U.load_source(ROOT.parent.parent/'diffusion_mmu/ckpt/source.pt').eval().requires_grad_(False)
    x,y=cifar10();x,y=x.cuda(),y.cuda();f=y==0;diff=Diffusion()
    for name,repel in [('frozen_redirect',False),('two_teacher',True)]:
        path=WORK/f'{name}.pt'
        if path.exists():continue
        student=copy.deepcopy(source).requires_grad_(True).train()
        torch.manual_seed(42);rb=cycle(x[~f],y[~f]);fb=cycle(x[f],y[f])
        opt=torch.optim.Adam(student.parameters(),lr=1e-4);hist=[];begin=time.time()
        for step in range(1000):
            xr,yr=next(rb);xf,yf=next(fb)
            tr=torch.randint(0,1000,(len(xr),),device='cuda');tf=torch.randint(0,1000,(len(xf),),device='cuda')
            nr=torch.randn_like(xr);nf=torch.randn_like(xf)
            xr=diff.q_sample(xr,tr,nr);xf=diff.q_sample(xf,tf,nf)
            with torch.no_grad():
                good=source(xf,tf,torch.ones_like(yf));bad=source(xf,tf,yf)
                targetr=source(xr,tr,yr)
            sf=student(xf,tf,yf);sr=student(xr,tr,yr)
            attraction=F.mse_loss(sf,good);distance=F.mse_loss(sf,bad)
            hinge=F.relu(.002-distance) if repel else sf.new_zeros(())
            denoise=F.mse_loss(sr,nr);preserve=F.mse_loss(sr,targetr)
            loss=attraction+hinge+3.072*(denoise+preserve)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.);assert torch.isfinite(norm)
            opt.step()
            if step%100==0 or step==999:
                row=dict(step=step,attraction=float(attraction.detach()),hinge=float(hinge.detach()),distance=float(distance.detach()),denoise=float(denoise.detach()),preservation=float(preserve.detach()))
                hist.append(row);print(name,row,flush=True)
        assert all(p.grad is None for p in source.parameters())
        assert all(torch.isfinite(p).all() for p in student.parameters())
        torch.save(dict(base=student.stem.out_channels,model=student.state_dict(),config=config),path)
        (OUT/f'{name}_training.json').write_text(json.dumps(dict(seconds=time.time()-begin,history=hist),indent=2))
    old=torch.load(ROOT/'samples.pt',map_location='cpu',weights_only=False)
    new=torch.load(ROOT/'expanded/new_samples.pt',map_location='cpu',weights_only=False)
    images={name:torch.cat([v.reshape(10,16,3,32,32),new['images'][name].reshape(10,16,3,32,32)],dim=1) for name,v in old['images'].items()}
    labels=torch.arange(10).repeat_interleave(16)
    for name in ['frozen_redirect','two_teacher']:
        path=WORK/f'{name}_samples.pt'
        if path.exists(): v=torch.load(path,weights_only=True)
        else:
            model=U.load_source(WORK/f'{name}.pt');batches=[]
            for s in [10042,10043]:
                torch.manual_seed(s);batches.append(generate(model,diff,labels,bs=160,guidance=2,steps=1000).reshape(10,16,3,32,32))
            v=torch.cat(batches,dim=1);assert torch.isfinite(v).all();torch.save(v,path)
        images[name]=v
    clf=judge();metrics={}
    for name,v in images.items():
        pred=classify(v.flatten(0,1).cuda(),clf).cpu().reshape(10,32)
        metrics[name]=dict(forget_airplane_count=int((pred[0]==0).sum()),forget_n=32,
            forget_airplane_rate=100*float((pred[0]==0).float().mean()),
            retain_correct=int((pred[1:]==torch.arange(1,10)[:,None]).sum()),retain_n=288,
            retain_cond_acc=100*float((pred[1:]==torch.arange(1,10)[:,None]).float().mean()))
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    rows=[('Source','Source'),('two_teacher','MMU'),
          ('Random mask (local)','Random mask'),('SalUn (local)','SalUn')]
    with PdfPages(OUT/'comparison.pdf') as pdf:
        for cls in range(10):
            fig,axes=plt.subplots(len(rows),8,figsize=(14,7))
            fig.subplots_adjust(left=.19,right=.995,top=.96,bottom=.02,hspace=.08,wspace=.04)
            for i,(key,label) in enumerate(rows):
                for j in range(8):
                    ax=axes[i,j];ax.imshow(((images[key][cls,16+j].permute(1,2,0)+1)/2).clamp(0,1),interpolation='nearest')
                    ax.set_xticks([]);ax.set_yticks([])
                    for s in ax.spines.values():s.set_visible(False)
                    color='#bd2026' if cls==0 else '#1a7f37'
                    if i==0:ax.set_title(f'{CLASSES[cls]} {17+j}',fontsize=10,color=color)
                    if i==0 and j==0:
                        ax.text(0,1.25,'Forgotten class: airplane' if cls==0 else f'Retained class: {CLASSES[cls]}   (forgotten class: airplane)',
                                transform=ax.transAxes,fontsize=14,fontweight='bold',color=color)
                    if j==0:ax.set_ylabel(label,fontsize=11)
            pdf.savefig(fig,dpi=220,bbox_inches='tight')
            fig.savefig(OUT/f'{CLASSES[cls]}.pdf',dpi=220,bbox_inches='tight')
            if cls==0:fig.savefig(OUT/'airplane.png',dpi=180,bbox_inches='tight')
            plt.close(fig)
    lines=['# CIFAR-10 two-teacher transfer: seed 42','',
      'Same source, forgotten airplane class, two evaluation-noise seeds and 1,000-step sampler as the existing expanded figure. One fixed training seed; no hyperparameter search. Full-width model only.', '',
      '| Method | D_f: airplane / 32 | D_f: airplane rate ↓ | D_r: condition accuracy ↑ |','|---|---:|---:|---:|']
    for name,r in metrics.items():lines.append(f'| {name} | {r["forget_airplane_count"]}/32 | {r["forget_airplane_rate"]:.2f} | {r["retain_cond_acc"]:.2f} |')
    lines+=['','The two new runs differ only by the repulsion term. Both use frozen-teacher attraction on noisy forgotten images, retained denoising and teacher preservation on retained data. The teacher uses automobile conditioning as the acceptable target and airplane conditioning as the unwanted reference. The margin is the fixed transferred SD value 0.002; its suitability for this model is not established. The hinge acts on batch-mean per-coordinate MSE.', '',
      'Each new run has 1,000 Adam updates, matching the existing SalUn/random update budget, but adds teacher preservation. Earlier MMU checkpoints used different training recipes. Lower airplane rate alone can reflect destroyed generation; inspect retained accuracy and images. The report uses only 32 forgotten and 288 retained generated images per method, so it is an exploratory comparison, not a reliable FID benchmark or a general superiority claim. Raw checkpoints and all sampled tensors are preserved in $MMU_WORK/mmu_cifar_two_teacher.', '',
      '[Comparison](comparison.pdf) · [Airplane](airplane.pdf) · [Settings](config.json) · [Raw metrics](metrics.json)','']
    (OUT/'report.md').write_text('\n'.join(lines))
    print('CIFAR TWO-TEACHER COMPLETE',metrics['two_teacher'],flush=True)

if __name__=='__main__':main()
