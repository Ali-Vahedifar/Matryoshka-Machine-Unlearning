import os
import copy,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
import generation_figure as B
from evaluate import generate,judge,classify,CLASSES
W=Path(os.environ.get('MMU_WORK','work')+'/mmu2t_strengthening/generation');W.mkdir(parents=True,exist_ok=True)
C=dict(seed=42,classes=[0,1,2],steps=1000,checkpoints=[100,250,500,1000],lr=.0001,batch=128,margin=.002,retain_weight=3.072,widths=[.5,.75,1.],sampling_steps=1000,samples_per_condition=16,sampling_seed=10043,heldout_seeds=[60042,60043],sequential_order=[0,1,2],sequential_target=9,baseline='existing local SalUn/random objective, exact 50% masks',nested='student prefix ablation; full-width frozen targets shared across widths')
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def train(source,initial,x,y,active,removed,method,wd,steps=1000,target=None):
 wd.mkdir(parents=True,exist_ok=True);schedule=[v for v in C['checkpoints'] if v<steps]+[steps]
 if all((wd/f'step{s}.pt').exists() for s in schedule):return
 torch.manual_seed(42);student=copy.deepcopy(initial).requires_grad_(True).train();teacher=copy.deepcopy(source).eval().requires_grad_(False)
 fm=torch.zeros_like(y,dtype=torch.bool);rm=fm.clone()
 for c in active:fm|=y==c
 for c in removed:rm|=y==c
 mask=None
 if method in ['SalUn','Random mask']:
  mp=wd/'mask.pt'
  if not mp.exists():torch.save(B.masks(source,x[fm],y[fm]),mp)
  masks=torch.load(mp,map_location='cuda',weights_only=True);mask=masks['salun' if method=='SalUn' else 'random'];del masks
 torch.manual_seed(42);rb=B.cycle(x[~rm],y[~rm]);fb=B.cycle(x[fm],y[fm]);diff=B.Diffusion();opt=torch.optim.Adam(student.parameters(),lr=.0001)
 history=[];begin=time.time();widths=C['widths'] if method.startswith('Nested') else [1.]
 for step in range(1,steps+1):
  xr,yr=next(rb);xf,yf=next(fb)
  tr=torch.randint(0,1000,(len(xr),),device='cuda');tf=torch.randint(0,1000,(len(xf),),device='cuda');nr=torch.randn_like(xr);nf=torch.randn_like(xf)
  xr=diff.q_sample(xr,tr,nr);xf=diff.q_sample(xf,tf,nf);yt=(yf+1)%10 if target is None else torch.full_like(yf,target)
  opt.zero_grad(set_to_none=True)
  if mask is not None:
   yr=yr.clone();yr[torch.rand(len(yr),device='cuda')<.1]=student.num_classes
   sr=student(xr,tr,yr)
   with torch.no_grad():good=student(xf,tf,yt)
   attraction=F.mse_loss(student(xf,tf,yf),good);retain=3.072*F.mse_loss(sr,nr);hinge=attraction.new_zeros(());loss=attraction+retain;loss.backward()
  else:
   with torch.no_grad():good=teacher(xf,tf,yt);bad=teacher(xf,tf,yf);rt=teacher(xr,tr,yr)
   vals=[]
   for width in widths:
    sf=student(xf,tf,yf,prefix_frac=width);sr=student(xr,tr,yr,prefix_frac=width)
    a=F.mse_loss(sf,good);dist=F.mse_loss(sf,bad);h=F.relu(.002-dist) if 'attraction' not in method.lower() else dist.new_zeros(())
    r=3.072*(F.mse_loss(sr,nr)+F.mse_loss(sr,rt));l=(a+h+r)/len(widths);l.backward();vals.append([a.item(),h.item(),r.item(),l.item()])
   attraction,hinge,retain,loss=np.array(vals).mean(0)
  norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.);assert torch.isfinite(norm)
  if mask is not None:
   for n,p in student.named_parameters():
    if p.grad is not None:p.grad.mul_(mask[n])
  opt.step()
  if step==1 or step%100==0 or step in schedule:
   history.append(dict(step=step,attraction=float(attraction),hinge=float(hinge),retain=float(retain),loss=float(loss),gradient_norm=float(norm),seconds=time.time()-begin));print(wd.name,method,history[-1],flush=True)
  if step in schedule:
   assert all(torch.isfinite(p).all() for p in student.parameters()) and all(p.grad is None for p in teacher.parameters())
   torch.save(dict(base=student.stem.out_channels,model=student.state_dict(),config={**C,'method':method,'active':active,'removed':removed,'target':target},seconds=time.time()-begin),wd/f'step{step}.pt')
   dump(wd/'history.json',history)

def evaluate(path,removed,seed,out,clf,target=None):
 sp=out.with_suffix('.pt');out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():return json.loads(out.read_text())
 if sp.exists():v=torch.load(sp,map_location='cpu',weights_only=True)
 else:
  model=B.U.load_source(path);torch.manual_seed(seed);v=generate(model,B.Diffusion(),torch.arange(10).repeat_interleave(16),bs=160,guidance=2,steps=1000).reshape(10,16,3,32,32);torch.save(v,sp);del model
 assert torch.isfinite(v).all();pred=classify(v.flatten(0,1).cuda(),clf).cpu().reshape(10,16)
 rates={str(c):dict(correct=int((pred[c]==c).sum()),replacement=int((pred[c]==((c+1)%10 if target is None else target)).sum()),n=16,predictions=pred[c].tolist()) for c in range(10)}
 keep=[c for c in range(10) if c not in removed];r=dict(checkpoint=str(path),seed=seed,removed=list(removed),per_class=rates,retain_correct=sum(rates[str(c)]['correct'] for c in keep),retain_n=16*len(keep))
 dump(out,r);return r

def main():
 torch.set_num_threads(4);dump(R/'generation_protocol.json',C)
 source=B.U.load_source(B.D/'ckpt/source.pt').eval().requires_grad_(True);x,y=B.cifar10();x,y=x.cuda(),y.cuda();clf=judge();results={}
 methods=['MMU','Attraction only','Nested MMU','Nested attraction','Random mask','SalUn']
 for c in C['classes']:
  for method in methods:
   key=f'ablation/class{c}/{method.replace(" ","_")}';wd=W/key
   train(source,source,x,y,[c],[c],method,wd)
   for s in C['checkpoints']:
    label=f'{key}/step{s}';results[label]=evaluate(wd/f'step{s}.pt',[c],10043,W/'evaluations'/f'{label}.json',clf)
   dump(R/'generation_results.json',results);print('ABLATION DONE',c,method,flush=True)
 for method in ['MMU','Random mask','SalUn']:
  initial=source;removed=[]
  for c in C['sequential_order']:
   removed.append(c);key=f'sequential/stage{len(removed)}/{method.replace(" ","_")}';wd=W/key
   train(source,initial,x,y,[c],removed,method,wd,target=9)
   path=wd/'step1000.pt';results[key]=evaluate(path,removed,10043,W/'evaluations'/f'{key}.json',clf,target=9)
   initial=B.U.load_source(path).eval();dump(R/'generation_results.json',results)
  del initial
  key=f'joint/{method.replace(" ","_")}';wd=W/key
  train(source,source,x,y,[0,1,2],[0,1,2],method,wd,steps=3000,target=9)
  results[key]=evaluate(wd/'step3000.pt',[0,1,2],10043,W/'evaluations'/f'{key}.json',clf,target=9);dump(R/'generation_results.json',results)
 for seed in C['heldout_seeds']:
  key=f'heldout/source/{seed}';results[key]=evaluate(B.D/'ckpt/source.pt',[],seed,W/'evaluations'/f'{key}.json',clf)
  for c in C['classes']:
   for method in ['MMU','Random mask','SalUn']:
    key=f'heldout/class{c}/{method.replace(" ","_")}/{seed}';path=W/f'ablation/class{c}'/method.replace(' ','_')/'step1000.pt'
    results[key]=evaluate(path,[c],seed,W/'evaluations'/f'{key}.json',clf);dump(R/'generation_results.json',results)
 dump(R/'generation_status.json',dict(status='complete',evaluations=len(results)));print('GENERATION STUDY COMPLETE',flush=True)
if __name__=='__main__':main()
