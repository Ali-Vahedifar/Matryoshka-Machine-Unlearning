import os
import copy, json, shutil, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import generation_figure as B
from evaluate import generate, judge, classify, CLASSES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
OUT=ROOT/'all_classes_seed42'
WORK=Path(os.environ.get('MMU_WORK','work')+'/mmu_cifar_all_classes')
ROWS=['Source','MMU','Random mask','SalUn']
CONFIG=dict(seed=42,mask_seed=4242,train_steps=1000,lr=1e-4,batch_size=128,margin=.002,mmu_retain_weight=3.072,mask_fraction=.5,sampling_seed=10043,sampling_steps=1000,guidance=2,samples_per_condition=16,display_sample_index=0,target_rule='(forgotten_class + 1) % 10',note='Local DDPM adaptation; full-width MMU with two frozen-source conditioned targets; no sweep.')

def mmu(source,x,y,f):
    teacher=copy.deepcopy(source).eval().requires_grad_(False)
    model=copy.deepcopy(source).requires_grad_(True).train()
    torch.manual_seed(42); rb=B.cycle(x[~f],y[~f]); fb=B.cycle(x[f],y[f]); diff=B.Diffusion()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4); hist=[]
    for step in range(1000):
        xr,yr=next(rb); xf,yf=next(fb)
        tr=torch.randint(0,1000,(len(xr),),device='cuda'); tf=torch.randint(0,1000,(len(xf),),device='cuda')
        nr=torch.randn_like(xr); nf=torch.randn_like(xf)
        xr=diff.q_sample(xr,tr,nr); xf=diff.q_sample(xf,tf,nf)
        with torch.no_grad():
            good=teacher(xf,tf,(yf+1)%10); bad=teacher(xf,tf,yf); targetr=teacher(xr,tr,yr)
        sf=model(xf,tf,yf); sr=model(xr,tr,yr)
        attraction=F.mse_loss(sf,good); distance=F.mse_loss(sf,bad)
        hinge=F.relu(.002-distance); denoise=F.mse_loss(sr,nr); preserve=F.mse_loss(sr,targetr)
        loss=attraction+hinge+3.072*(denoise+preserve)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); assert torch.isfinite(norm)
        opt.step()
        if step%100==0 or step==999:
            row=dict(step=step,loss=loss.item(),attraction=attraction.item(),hinge=hinge.item(),preserve=preserve.item())
            hist.append(row); print('MMU',row,flush=True)
    assert all(p.grad is None for p in teacher.parameters())
    return model,hist

def page(grids,c):
    fig,axes=plt.subplots(4,10,figsize=(16,6.2))
    fig.subplots_adjust(left=.135,right=.995,top=.8,bottom=.02,hspace=.05,wspace=.05)
    red,green='#bd2026','#1a7f37'
    for r,name in enumerate(ROWS):
        for k in range(10):
            ax=axes[r,k]; ax.imshow(((grids[name][k,0].permute(1,2,0)+1)/2).clamp(0,1),interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(k==c); s.set_edgecolor('#bd2026'); s.set_linewidth(2)
            if r==0:
                ax.set_title(f'{CLASSES[k]}\n'+('FORGET' if k==c else 'retain'),fontsize=14,fontweight='bold',color=red if k==c else green)
                if k==0: ax.text(0,1.5,f'Forgotten class: {CLASSES[c]}',transform=ax.transAxes,fontsize=20,fontweight='bold',color=red)
                if k==9: ax.text(1,1.5,'Retained: the other 9 classes',transform=ax.transAxes,ha='right',fontsize=20,fontweight='bold',color=green)
            if k==0: ax.set_ylabel(name.replace(' ','\n'),fontsize=16,fontweight='bold')
    return fig

def report(metrics,complete=False):
    lines=['# One class removed per experiment: CIFAR-10, seed 42','',f'Status: {"complete" if complete else "running"}; {len(metrics)}/10 forgotten classes evaluated.','',
      'Each page is a separately unlearned model, initialized from the same source. Rows: Source, MMU, Random mask, SalUn. Columns show all ten generation conditions. Red outlines identify the forgotten condition, not detector results. The same initial noise is used across methods and experiments. The displayed sample is fixed in advance (index 0); no selection by appearance.', '',
      'One training seed (42), 1,000 updates per method and deletion class. Class-0 checkpoints and matched samples are reused; the other nine deletions are trained independently from the source. MMU uses full-width frozen-source attraction and margin repulsion; its alternative target condition is (forgotten class + 1) modulo 10. SalUn and Random mask use the previous local recipes with class-specific 50% masks. This is not an official SalUn reproduction.', '',
      'Exploratory classifier metrics use 16 generated samples per condition (16 forgotten and 144 retained). D_f is the fraction classified as the forgotten class under its condition; lower alone does not establish successful unlearning. D_r is retained-condition accuracy. No retrain reference, FID, or multiple-seed significance is established here. Native images are 32×32; nearest-neighbor rendering preserves their actual detail.', '',
      '| Forgotten class | Method | D_f condition accuracy (%) | D_r condition accuracy (%) |','|---|---|---:|---:|']
    for c,rows in metrics.items():
        for name in ROWS:
            r=rows[name]; lines.append(f'| {CLASSES[int(c)]} | {name} | {100*r["forget_correct"]/16:.2f} | {100*r["retain_correct"]/144:.2f} |')
    lines += ['', '[Ten-page comparison](comparison.pdf) · [Configuration](config.json) · [Counts](metrics.json)', '',f'Checkpoints and raw samples: `{WORK}`.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))

def main():
    OUT.mkdir(exist_ok=True); WORK.mkdir(exist_ok=True); torch.set_num_threads(4)
    (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    source=B.U.load_source(B.D/'ckpt/source.pt').eval().requires_grad_(True)
    x,y=B.cifar10(); x,y=x.cuda(),y.cuda(); clf=judge()
    cached=torch.load(ROOT/'expanded/new_samples.pt',map_location='cpu',weights_only=False)['images']
    source_samples=cached['Source'].reshape(10,16,3,32,32)
    metrics={}; labels=torch.arange(10).repeat_interleave(16)
    for c in range(10):
        print('START CLASS',c,CLASSES[c],flush=True); f=y==c; wd=WORK/f'class_{c}'; wd.mkdir(exist_ok=True)
        paths={'MMU':wd/'mmu.pt','Random mask':wd/'random.pt','SalUn':wd/'salun.pt'}
        if c==0:
            paths={'MMU':Path(os.environ.get('MMU_WORK','work')+'/mmu_cifar_two_teacher/two_teacher.pt'),'Random mask':ROOT/'random.pt','SalUn':ROOT/'salun.pt'}
        if not all(p.exists() for p in paths.values()):
            mask_path=wd/'masks.pt'
            if not mask_path.exists(): torch.save(B.masks(source,x[f],y[f]),mask_path)
            masks=torch.load(mask_path,map_location='cuda',weights_only=True)
            for name,path in paths.items():
                if path.exists(): continue
                start=time.time(); print('TRAIN',c,name,flush=True)
                if name=='MMU': model,hist=mmu(source,x,y,f)
                else: model,hist=B.train_masked(source,masks['salun' if name=='SalUn' else 'random'],x[~f],y[~f],x[f],y[f])
                assert all(torch.isfinite(p).all() for p in model.parameters())
                torch.save(dict(base=model.stem.out_channels,model=model.state_dict(),config={**CONFIG,'forgotten_class':c},history=hist,seconds=time.time()-start),path)
                print('TRAIN DONE',c,name,time.time()-start,flush=True); del model
            del masks
        grids={'Source':source_samples}
        for name,path in paths.items():
            sp=wd/(path.stem+'_samples.pt')
            if sp.exists(): v=torch.load(sp,map_location='cpu',weights_only=True)
            elif c==0:
                if name=='MMU': v=torch.load(os.environ.get('MMU_WORK','work')+'/mmu_cifar_two_teacher/two_teacher_samples.pt',map_location='cpu',weights_only=True)[:,16:32]
                else: v=cached['SalUn (local)' if name=='SalUn' else 'Random mask (local)'].reshape(10,16,3,32,32)
                torch.save(v,sp)
            else:
                print('SAMPLE',c,name,flush=True); model=B.U.load_source(path)
                torch.manual_seed(10043)
                v=generate(model,B.Diffusion(),labels,bs=160,guidance=2,steps=1000).reshape(10,16,3,32,32)
                assert torch.isfinite(v).all(); torch.save(v,sp); del model
            grids[name]=v
        rows={}
        for name,v in grids.items():
            pred=classify(v.flatten(0,1).cuda(),clf).cpu().reshape(10,16)
            correct=pred==torch.arange(10)[:,None]; retained=torch.arange(10)!=c
            rows[name]=dict(forget_correct=int(correct[c].sum()),forget_n=16,retain_correct=int(correct[retained].sum()),retain_n=144)
        metrics[str(c)]=rows; report(metrics)
        fig=page(grids,c); fig.savefig(OUT/f'forget_{c}_{CLASSES[c]}.pdf',bbox_inches='tight',dpi=200); plt.close(fig)
        print('CLASS COMPLETE',c,rows,flush=True)
    with PdfPages(OUT/'comparison.pdf') as pdf:
        for c in range(10):
            wd=WORK/f'class_{c}'; grids={'Source':source_samples}
            for name,stem in [('MMU','two_teacher' if c==0 else 'mmu'),('Random mask','random'),('SalUn','salun')]:
                grids[name]=torch.load(wd/f'{stem}_samples.pt',map_location='cpu',weights_only=True)
            fig=page(grids,c); pdf.savefig(fig,bbox_inches='tight',dpi=200); plt.close(fig)
    report(metrics,True)
    target=ROOT/'expanded/comparison.pdf'; backup=ROOT/'expanded/comparison_before_all_class_deletions.pdf'
    if target.exists() and not backup.exists(): shutil.copy2(target,backup)
    shutil.copy2(OUT/'comparison.pdf',target)
    (ROOT/'expanded/README.md').write_text('The main comparison.pdf now contains ten independent class-deletion experiments, with Source, MMU, Random mask, and SalUn only. See ../all_classes_seed42/report.md for results and settings. Earlier numerical results and raw samples are preserved.\n')
    print('ALL TEN CLASSES COMPLETE',flush=True)

if __name__=='__main__': main()
