import os
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent));import classification as C
W=Path(os.environ.get('MMU_WORK','work')+'/mmu2t_strengthening/classification');W.mkdir(parents=True,exist_ok=True)
def dump(p,v):p.write_text(json.dumps(v,indent=2))
def kl(s,t):return F.kl_div(F.log_softmax(s,1),F.softmax(t,1),reduction='batchmean')
def cycle(dl):
 while True:yield from dl

def main():
 torch.set_num_threads(4);sys.argv=['classification_study'];args=C.arg_parser.parse_args();args.seed=args.train_seed=2;args.data=os.environ.get('MMU_DATA','data');args.print_freq=10000
 orig=C.utils.cifar10_dataloaders
 def same(*a,**kw):kw.setdefault('seed',2);return orig(*a,**kw)
 C.utils.cifar10_dataloaders=same
 source,tr,va,te,_=C.utils.setup_model_dataset(args);source=source.cuda();source.load_state_dict(torch.load(R.parent/'classification/seed2/source.pt',weights_only=False)['state_dict']);source.eval()
 cfg=dict(seed=2,ratios=[.1,.5],epochs=5,lr=.001,margin=.002,temperature=1,good_teacher='source fine-tuned on retained data for 5 epochs, SGD lr .01',bad_teacher='frozen source',note='Classification adaptation, not identical to conditioned diffusion. Full width; teacher construction time included. No test tuning; seed 2 reuses paired source and retrain. SCRUB control is local full-width max/min objective, not an official reproduction.')
 dump(R/'classification_protocol.json',cfg);results={}
 for ratio in [10,50]:
  base=R.parent/f'classification/seed2/forget{ratio}';wd=W/f'forget{ratio}';wd.mkdir(exist_ok=True);split=np.load(base/'split.npz')
  def take(idx):
   ds=copy.deepcopy(tr.dataset);ds.data=ds.data[idx];ds.targets=np.asarray(ds.targets)[idx];return ds
  forget=take(np.sort(split['forget']));retain=take(split['retain']);rd=C.loader(retain,True);fd=C.loader(forget,True)
  gp=wd/'good_teacher.pt'
  if not gp.exists():
   C.utils.setup_seed(2);good=copy.deepcopy(source).train();opt=torch.optim.SGD(good.parameters(),lr=.01,momentum=.9,weight_decay=.0005);start=time.time()
   for epoch in range(5):
    for x,y in rd:
     loss=F.cross_entropy(good(x.cuda()),y.cuda());opt.zero_grad();loss.backward();opt.step()
    print('GOOD TEACHER',ratio,epoch,flush=True)
   torch.save(dict(state_dict=good.state_dict(),seconds=time.time()-start),gp)
  obj=torch.load(gp,weights_only=False);teacher_seconds=obj['seconds'];good=copy.deepcopy(source).eval().requires_grad_(False);good.load_state_dict(obj['state_dict']);bad=copy.deepcopy(source).eval().requires_grad_(False)
  paths={'Source':R.parent/'classification/seed2/source.pt','Retrain':base/'retrain.pt','SalUn':base/'SalUn.pt','Retain FT teacher':gp}
  for method in ['MMU','Attraction only','Full-width SCRUB control']:
   path=wd/(method.replace(' ','_')+'.pt');paths[method]=path
   if path.exists():continue
   C.utils.setup_seed(2);model=copy.deepcopy(source).train();opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005);start=time.time();history=[]
   for epoch in range(5):
    model.train();total=0.;n=0
    if method=='Full-width SCRUB control':
     if epoch<2:
      for x,y in fd:
       x=x.cuda()
       with torch.no_grad():bt=bad(x)
       loss=-kl(model(x)/4,bt/4)*16;opt.zero_grad();loss.backward();opt.step()
     for x,y in rd:
      x,y=x.cuda(),y.cuda()
      with torch.no_grad():bt=bad(x)
      s=model(x);loss=F.cross_entropy(s,y)+16*kl(s/4,bt/4);opt.zero_grad();loss.backward();opt.step();total+=loss.item();n+=1
    else:
     ff=cycle(fd)
     for xr,yr in rd:
      xf,_=next(ff);xr,yr,xf=xr.cuda(),yr.cuda(),xf.cuda()
      with torch.no_grad():gt=good(xf);bt=bad(xf);rt=bad(xr)
      sf=model(xf);sr=model(xr);attraction=kl(sf,gt);hinge=F.relu(.002-kl(sf,bt)) if method=='MMU' else sf.new_zeros(())
      loss=attraction+hinge+F.cross_entropy(sr,yr)+kl(sr,rt);assert torch.isfinite(loss)
      opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item();n+=1
    history.append(dict(epoch=epoch,loss=total/n,seconds=time.time()-start));print('CLASSIFICATION',ratio,method,history[-1],flush=True)
   assert all(p.grad is None for p in good.parameters()) and all(p.grad is None for p in bad.parameters())
   assert all(torch.isfinite(p).all() for p in model.parameters())
   torch.save(dict(state_dict=model.state_dict(),seconds=time.time()-start,teacher_seconds=teacher_seconds if method!='Full-width SCRUB control' else 0,history=history),path)
  rr={}
  for method,path in paths.items():
   ep=wd/(method.replace(' ','_')+'.json')
   if ep.exists():rr[method]=json.loads(ep.read_text());continue
   obj=torch.load(path,weights_only=False);model=copy.deepcopy(source);model.load_state_dict(obj['state_dict']);model.eval();r=C.evaluate(model,retain,forget,te.dataset)
   r['Df_accuracy']=100-r['UA'];r['Dr_accuracy']=r['RA'];r['Test_accuracy']=r['TA'];r['seconds']=obj.get('seconds',None);r['teacher_seconds']=obj.get('teacher_seconds',0);r['checkpoint']=str(path)
   dump(ep,r);rr[method]=r;print('METRICS',ratio,method,r,flush=True)
  ref=rr['Retrain']
  for r in rr.values():r['avg_gap']=sum(abs(r[k]-ref[k]) for k in ['UA','RA','TA','MIA'])/4
  results[str(ratio)]=rr;dump(R/'classification_results.json',results)
 dump(R/'classification_status.json',dict(status='complete'));print('CLASSIFICATION COMPLETE',flush=True)
if __name__=='__main__':main()
