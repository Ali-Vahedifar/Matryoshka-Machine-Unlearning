import os
import copy,importlib.util,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parent;PROJECT=R.parents[2];W=Path(os.environ.get('MMU_WORK','work'))/'mmu_variants_3seed';W.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(PROJECT/'mmu'),str(PROJECT/'mmu/scripts')]
import benchmark_unlearning as B
spec=importlib.util.spec_from_file_location('variant_mmu',PROJECT/'mmu/MMU/mmu.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
V=[]
for objective in ['unbounded','margin']:
 for nest,head,weight in [(False,'tied','mrl'),(True,'independent','mrl'),(True,'independent','normalised'),(True,'tied','mrl'),(True,'tied','normalised')]:
  name=f'MMU {objective} / '+('full' if not nest else f'nested {"MRL" if head=="independent" else "MRL-E"} / {"sum" if weight=="mrl" else "weighted mean"}')
  V.append(dict(name=name,family='source_roles',nest=nest,head=head,weight=weight,margin=None if objective=='unbounded' else .05,teacher='width-aligned source',attraction_forget=False))
V.append(dict(name='MMU margin / nested MRL / full privileged teacher',family='source_roles',nest=True,head='independent',weight='normalised',margin=.05,teacher='full source',attraction_forget=False))
for nest in [False,True]:
 for repel in [False,True]:
  V.append(dict(name='MMU retain-FT two-network / '+('nested MRL' if nest else 'full')+' / '+('attraction + margin' if repel else 'attraction only'),family='retain_ft_adapter',nest=nest,head='independent',weight='normalised',margin=.002 if repel else None,teacher='full retain-FT + full source',attraction_forget=True))
for i,v in enumerate(V,1):v['id']=f'V{i:02}'
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
class CachedBenchmark(B.Benchmark):
 def _source(self):
  if not self._cache_matches():raise RuntimeError('Cached source protocol mismatch: refusing to overwrite prior experiments')
  state=torch.load(self.cache/'source.pt',map_location='cpu',weights_only=False)
  manifest=json.loads((self.cache/'manifest.json').read_text())
  assert B.state_digest(state)==manifest['source_sha256']
  return state,{}
 def _reference(self):
  assert self._cache_matches();meta=json.loads((self.cache/'manifest.json').read_text())
  state=torch.load(self.cache/'retrain_reference.pt',map_location='cpu',weights_only=False)
  if meta.get('retrain_sha256'):assert B.state_digest(state)==meta['retrain_sha256']
  return state,meta['retrain_time']
def setup(mode,seed):
 old=Path(os.environ.get('MMU_WORK','work'))/f'cifar10/mu/resnet18_adam_{mode}/seed{seed}';out=W/mode/f'seed{seed}'
 sys.argv=['variants','--dataset','cifar10','--backbone','resnet18','--optimizer','adam','--epochs','100','--early_stopping_patience','10','--min_epochs','20','--batch_size','64','--source_lr','.001','--retain_ratio','.1','--selection','utility','--utility_tolerance','2','--relearn_epochs','5','--seed',str(seed),'--device','cuda:0','--data_dir',os.environ.get('MMU_DATA','data'),'--output_dir',str(out),'--cache_dir',str(old/'cache'),'--forget_mode',mode,'--methods','baseline,retrain,mmu,salun']
 sys.argv+=['--num_forget','4500'] if mode=='instance' else ['--forget_class','0']
 a=B.parse_args();b=CachedBenchmark(a);return b,old,out

def teacher_fit(b,out):
 p=out/'retain_ft_teacher.pt';model=b._fresh_source()
 if p.exists():o=torch.load(p,weights_only=False);model.load_state_dict(o['model']);return model.eval().requires_grad_(False),o['seconds']
 B.set_seed(b.args.seed);start=time.time();opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0005);model.train()
 for _ in range(5):
  for x,y in b.loaders['retain_small']:
   loss=F.cross_entropy(model(x.to(b.device)),y.to(b.device));opt.zero_grad();loss.backward();opt.step()
 torch.cuda.synchronize();sec=time.time()-start;torch.save(dict(model=model.state_dict(),seconds=sec),p)
 return model.eval().requires_grad_(False),sec

def custom(b,v,hp,good=None):
 model=b._fresh_source();widths=list(b.mmu_granularities) if v['nest'] else [M.linear_head(model).in_features]
 method=M.MSCRUB(model,device=b.device,granularities=widths,head_mode=v['head'],weighting=v['weight'],diagnose=False,**hp)
 method.heads=M.matryoshka_heads(method.head,widths,b.device) if v['head']=='independent' else None
 pars=list(model.parameters())+[p for h in (method.heads or {}).values() for p in h.parameters()]
 teacher=copy.deepcopy(model).eval().requires_grad_(False);head=M.linear_head(teacher);a,beta=method._scales()
 opt=torch.optim.SGD(pars,lr=method.lr,momentum=.9,weight_decay=.0005);history=[]
 def targets(x):
  with torch.no_grad():
   if v['teacher']=='full source':return [teacher(x)]*len(widths)
   return M.nested_logits(teacher,head,x,widths)
 def cycle(dl):
  while True:yield from dl
 for epoch in range(method.epochs):
  model.train();total=0.;n=0
  if v['family']=='source_roles':
   if epoch<method.msteps:
    for x,_ in b.loaders['forget_train']:
     x=x.to(b.device);tt=targets(x);ss=method._student_logits(x)
     loss=sum(w*(-M.kd_kl(s,t,4.) if v['margin'] is None else F.relu(v['margin']-M.kd_kl(s,t,4.))) for w,s,t in zip(a,ss,tt))
     assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pars,5.);opt.step()
   for x,y in b.loaders['retain_small']:
    x,y=x.to(b.device),y.to(b.device);tt=targets(x);ss=method._student_logits(x)
    loss=sum(w*(F.cross_entropy(s,y)+M.kd_kl(s,t,4.)) for w,s,t in zip(beta,ss,tt));assert torch.isfinite(loss)
    opt.zero_grad();loss.backward();opt.step();total+=loss.item();n+=1
  else:
   ff=cycle(b.loaders['forget_train'])
   for xr,yr in b.loaders['retain_small']:
    xf,_=next(ff);xf,xr,yr=xf.to(b.device),xr.to(b.device),yr.to(b.device)
    with torch.no_grad():gt=good(xf);bt=teacher(xf);rt=teacher(xr)
    sf=method._student_logits(xf);sr=method._student_logits(xr)
    loss=sum(M.kd_kl(s,gt,1.)+(F.relu(v['margin']-M.kd_kl(s,bt,1.)) if v['margin'] is not None else s.new_zeros(()))+F.cross_entropy(r,yr)+M.kd_kl(r,rt,1.) for s,r in zip(sf,sr))/len(widths)
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pars,1.);opt.step();total+=loss.item();n+=1
  history.append(dict(epoch=epoch,retain_or_joint_loss=total/n));print(v['id'],epoch,total/n,flush=True)
 assert all(torch.isfinite(p).all() for p in pars) and all(p.grad is None for p in teacher.parameters())
 auxiliary={str(w):h.state_dict() for w,h in (method.heads or {}).items()}
 return model,history,auxiliary

def main():
 torch.set_num_threads(4);dump(R/'variants.json',V);all_results={}
 for mode in ['class','instance','subclass']:
  for seed in [42,43,44]:
   cell=f'{mode}/seed{seed}';dump(R/'status.json',dict(status='running',cell=cell));b,old,out=setup(mode,seed)
   best=json.loads((old/'search.json').read_text())['best'];core={k:best['mmu']['config'][k] for k in ['lr','epochs','msteps']};core['max_grad_norm']=5.
   dump(out/'protocol.json',dict(protocol=b.protocol,source_sha256=B.state_digest(b.source_state),variants=V,base_schedule=core,salun=best['salun']['config'],note='Prior validation-selected base schedule shared by source-role variants; no new test tuning. Adapter uses lr .001, five epochs, same 10% retained subset, and charges five-epoch teacher construction. Cached matching source/retrain reused, unlearning rerun.'))
   rows={};good=None;teacher_sec=0
   jobs=[dict(id='Source',name='Source'),dict(id='Retrain',name='Retrain'),dict(id='SalUn',name='SalUn (local)')]+V
   for v in jobs:
    id=v['id'];result=out/f'{id}.json';ck=out/f'{id}.pt'
    if result.exists():rows[id]=json.loads(result.read_text());continue
    B.set_seed(seed);start=time.time();extra=0.;hist=[];aux={};config={}
    if id=='Source':model=b._fresh_source();seconds=0.
    elif id=='Retrain':model=copy.deepcopy(b.reference);seconds=b.retrain_time
    elif ck.exists():
     obj=torch.load(ck,weights_only=False);model=b._fresh_source();model.load_state_dict(obj['model']);seconds=obj['seconds'];extra=obj.get('teacher_seconds',0);config=obj['config']
    elif id=='SalUn':
     config=best['salun']['config'];method=b._build('salun',b._fresh_source(),config);model=b._apply('salun',method,config);torch.cuda.synchronize();seconds=time.time()-start
    else:
     config={**core,'head_mode':v['head'],'weighting':v['weight'],'bad_margin':v['margin']}
     if v['family']=='retain_ft_adapter':
      if good is None:good,teacher_sec=teacher_fit(b,out)
      extra=teacher_sec;B.set_seed(seed);start=time.time();hp=dict(lr=.001,epochs=5,msteps=2,max_grad_norm=1.);config.update(hp)
     else:hp=core
     model,hist,aux=custom(b,v,hp,good);torch.cuda.synchronize();seconds=time.time()-start
    if id not in ['Source','Retrain'] and not ck.exists():torch.save(dict(model=model.state_dict(),auxiliary_heads=aux,seconds=seconds,teacher_seconds=extra,config=config,history=hist),ck)
    model.eval();metrics=b.evaluate(model,'final');assert B.finite_metrics(metrics)
    row=dict(id=id,name=v['name'],mode=mode,seed=seed,metrics=metrics,seconds=seconds,teacher_seconds=extra,config=config);dump(result,row);rows[id]=row
    print('RESULT',cell,id,metrics,flush=True);all_results[cell]=rows;dump(R/'results.json',all_results)
   all_results[cell]=rows;dump(R/'results.json',all_results);del b,good
 dump(R/'status.json',dict(status='complete',cells=9,methods=len(V)+3));print('ALL VARIANTS COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(R/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise
