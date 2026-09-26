import copy,json,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;D=ROOT.parent/'diffusion_mmu'
sys.path.insert(0,str(D))
import unlearn as U
from ddpm import Diffusion
from train_source import cifar10
from evaluate import generate,CLASSES
OUT=ROOT/'generation';OUT.mkdir(exist_ok=True)

def cycle(x,y):
    while True:
        yield from U.batches(x,y,128)


def masks(source,xf,yf):
    model=copy.deepcopy(source).eval();acc={n:torch.zeros_like(p) for n,p in model.named_parameters()}
    diff=Diffusion();torch.manual_seed(42)
    for x,y in U.batches(xf,yf,128):
        t=torch.randint(0,1000,(len(x),),device='cuda');e=torch.randn_like(x)
        model.zero_grad(set_to_none=True)
        F.mse_loss(model(diff.q_sample(x,t,e),t,y),e).backward()
        for n,p in model.named_parameters():
            if p.grad is not None:acc[n]+=p.grad
    flat=torch.cat([v.abs().flatten() for v in acc.values()]);mask=torch.zeros_like(flat)
    mask[torch.argsort(flat,descending=True)[:len(flat)//2]]=1
    g=torch.Generator(device='cuda').manual_seed(4242)
    random=mask[torch.randperm(len(mask),generator=g,device='cuda')]
    out={kind:{} for kind in ['salun','random']};start=0
    for n,p in model.named_parameters():
        for kind,m in [('salun',mask),('random',random)]:out[kind][n]=m[start:start+p.numel()].reshape_as(p).clone()
        start+=p.numel()
    assert sum(int(v.sum()) for v in out['salun'].values())==sum(int(v.sum()) for v in out['random'].values())
    return out


def train_masked(source,mask,xr,yr,xf,yf):
    torch.manual_seed(42);model=copy.deepcopy(source).train();diff=Diffusion()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4);rb=cycle(xr,yr);fb=cycle(xf,yf);hist=[]
    for step in range(1000):
        x,y=next(rb);t=torch.randint(0,1000,(len(x),),device='cuda');e=torch.randn_like(x)
        y=y.clone();y[torch.rand(len(y),device='cuda')<.1]=model.num_classes
        lr=(model(diff.q_sample(x,t,e),t,y)-e).square().flatten(1).sum(1).mean()
        x,y=next(fb);t=torch.randint(0,1000,(len(x),),device='cuda');e=torch.randn_like(x);xt=diff.q_sample(x,t,e)
        with torch.no_grad():target=model(xt,t,(y+1)%10)
        lf=F.mse_loss(model(xt,t,y),target);loss=lf+1e-3*lr
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        for n,p in model.named_parameters():
            if p.grad is not None:p.grad.mul_(mask[n])
        opt.step()
        if step%100==0:
            hist.append({'step':step,'retain_sum_mse':float(lr.detach()),'forget_mse':float(lf.detach())})
            print(hist[-1],flush=True)
    return model,hist


def main():
    config={'source':str(D/'ckpt/source.pt'),'seed':42,'mask_seed':4242,'forget_class':0,
            'train_steps':1000,'sampling_steps':1000,'guidance':2,'mask_fraction':.5,
            'retain_weight':.001,'retain_reduction':'sum over pixels, mean over batch',
            'forget_reduction':'per-coordinate mean','samples_per_class':16,
            'status':'local architecture adaptation; not released DDPM reproduction',
            'random':'exactly matched global mask cardinality; independent mask seed'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    source=U.load_source(D/'ckpt/source.pt');x,y=cifar10();x,y=x.cuda(),y.cuda();f=y==0
    checkpoint={
      'Source':D/'ckpt/source.pt','Retrain reference':D/'ckpt/retrain_c0.pt',
      'SalUn (local)':OUT/'salun.pt','Random mask (local)':OUT/'random.pt',
      'MMU full width, beta=1':D/'results/full_redirect_20260907_c0_seed42/mmu_fullwidth.pt',
      'MMU full width, beta=4':D/'results/improve_b4_c0_seed42/mmu_fullwidth.pt',
      'MMU nested, warm-up=2':D/'results/improve_warm2_c0_seed42/mmu.pt'}
    if not all((OUT/f'{k}.pt').exists() for k in ['salun','random']):
        mask=masks(source,x[f],y[f]);torch.save(mask,OUT/'masks.pt')
        for name in ['salun','random']:
            if (OUT/f'{name}.pt').exists():continue
            start=time.time();model,hist=train_masked(source,mask[name],x[~f],y[~f],x[f],y[f])
            torch.save({'base':model.stem.out_channels,'model':model.state_dict(),'config':config,
                        'history':hist,'seconds':time.time()-start},OUT/f'{name}.pt')
    grids={}
    labels=torch.arange(10).repeat_interleave(16)
    for name,path in checkpoint.items():
        print('GENERATING',name,flush=True);model=U.load_source(path)
        torch.manual_seed(10042)
        grids[name]=generate(model,Diffusion(),labels,bs=160,guidance=2,steps=1000)
        torch.save({'labels':labels,'images':grids},OUT/'samples.pt')
    indices=[0,1,2,3]+[c*16 for c in range(1,10)]
    fig,axes=plt.subplots(len(grids),13,figsize=(15,1.15*len(grids)),squeeze=False)
    for i,(name,images) in enumerate(grids.items()):
        for j,idx in enumerate(indices):
            ax=axes[i,j];ax.imshow(((images[idx].permute(1,2,0)+1)/2).clamp(0,1),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(f'I{j+1}' if j<4 else f'C{j-3}\n{CLASSES[j-3]}',fontsize=8)
            if j==0:ax.set_ylabel(name,fontsize=8)
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(OUT/f'figure4_comparison.{ext}',dpi=200,bbox_inches='tight')
    print('FIGURE COMPLETE',flush=True)

if __name__=='__main__':main()
