import os
import argparse,copy,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1];UP=REPO/'salun_official/Classification'
sys.path.insert(0,str(UP))
sys.path.append(str(REPO/'mmu'))
import arg_parser,utils,unlearn
from trainer import train
from evaluation.SVC_MIA import collect_prob,SVC_fit_predict


def dump(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)


def loader(ds,shuffle=False):
    return DataLoader(ds,batch_size=256,shuffle=shuffle,num_workers=0,pin_memory=True)


def evaluate(model,retain,forget,test):
    probs={};acc={}
    for name,ds in [('retain',retain),('forget',forget),('test',test)]:
        ds=copy.deepcopy(ds);ds.transform=transforms.ToTensor()
        p,y=collect_prob(loader(ds),model)
        acc[name]=100*float((p.argmax(1)==y).float().mean())
        probs[name]=p.gather(1,y[:,None])
    n=len(test)
    mia=100*float(SVC_fit_predict(probs['retain'][:n],probs['test'],
                   torch.zeros((0,1)),probs['forget']))
    return dict(UA=100-acc['forget'],RA=acc['retain'],TA=acc['test'],MIA=mia)


def mask_for(model,forget,fraction=.5):
    model.eval();grads={n:torch.zeros_like(p) for n,p in model.named_parameters()}
    for x,y in loader(forget,True):
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x.cuda()),y.cuda()).backward()
        for n,p in model.named_parameters():
            if p.grad is not None:grads[n].add_(p.grad)
    flat=torch.cat([g.abs().flatten() for g in grads.values()])
    order=torch.argsort(flat,descending=True);m=torch.zeros_like(flat)
    m[order[:int(len(flat)*fraction)]]=1
    result={};start=0
    for n,p in model.named_parameters():
        result[n]=m[start:start+p.numel()].reshape_as(p);start+=p.numel()
    model.zero_grad(set_to_none=True)
    return result


def fit(model,ds,val,args,path):
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=182)
    begin=time.perf_counter();hist=[];start=0
    partial=path.with_name(path.stem+'_progress.pt')
    if partial.exists():
        ck=torch.load(partial,weights_only=False);model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        hist=ck['history'];start=ck['epoch']+1
        torch.set_rng_state(ck['rng_cpu']);torch.cuda.set_rng_state(ck['rng_cuda'])
        np.random.set_state(ck['rng_numpy'])
        import random;random.setstate(ck['rng_python'])
        begin-=ck['seconds']
    for epoch in range(start,182):
        train(loader(ds,True),model,nn.CrossEntropyLoss(),opt,epoch,args)
        sched.step();hist.append({'epoch':epoch,'seconds':time.perf_counter()-begin})
        print(path.name,'epoch',epoch,'seconds',hist[-1]['seconds'],flush=True)
        if epoch%10==0 or epoch==181:
            import random
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                       epoch=epoch,history=hist,seconds=hist[-1]['seconds'],rng_cpu=torch.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state(),rng_numpy=np.random.get_state(),rng_python=random.getstate()),partial)
    torch.save({'state_dict':model.state_dict(),'history':hist,'seconds':time.perf_counter()-begin},path)
    if partial.exists():partial.unlink()
    return time.perf_counter()-begin


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--ratio',type=float,choices=[.1,.5],required=True)
    ap.add_argument('--methods',default='retrain,FT,RL,GA,IU,BE,BS,l1_sparse,SalUn,SalUn_soft,MMU')
    ap.add_argument('--source_only',action='store_true')
    ap.add_argument('--tune',action='store_true')
    ap.add_argument('--settings',type=Path)
    a=ap.parse_args()
    sys.argv=[sys.argv[0]];args=arg_parser.parse_args()
    args.seed=args.train_seed=a.seed;args.data=os.environ.get('MMU_DATA','data');args.print_freq=10000
    args.mask_ratio=.5
    args.save_dir=str(ROOT/f'classification/seed{a.seed}');Path(args.save_dir).mkdir(parents=True,exist_ok=True)
    original=utils.cifar10_dataloaders
    def same_split(*v,**kw):
        kw.setdefault('seed',a.seed);return original(*v,**kw)
    utils.cifar10_dataloaders=same_split
    model,tr,va,te,_=utils.setup_model_dataset(args);model=model.cuda()
    initial=copy.deepcopy(model.state_dict())
    source=Path(args.save_dir)/'source.pt'
    if not source.exists():fit(model,tr.dataset,va.dataset,args,source)
    else:model.load_state_dict(torch.load(source,weights_only=False)['state_dict'])
    progress=source.with_name('source_progress.pt')
    if source.exists() and progress.exists():progress.unlink()
    if a.source_only:return
    out=Path(args.save_dir)/f'forget{int(a.ratio*100)}';out.mkdir(exist_ok=True)
    n=len(tr.dataset);rng=np.random.RandomState(a.seed-1)
    chosen=rng.choice(n,int(n*a.ratio),replace=False);forget_flag=np.zeros(n,dtype=bool);forget_flag[chosen]=True
    np.savez(out/'split.npz',forget=chosen,retain=np.flatnonzero(~forget_flag))
    forget=copy.deepcopy(tr.dataset);retain=copy.deepcopy(tr.dataset)
    forget.data=forget.data[forget_flag];forget.targets=np.asarray(forget.targets)[forget_flag]
    retain.data=retain.data[~forget_flag];retain.targets=np.asarray(retain.targets)[~forget_flag]
    data={'forget':loader(forget,True),'retain':loader(retain,True),'test':te,'val':va}
    source_state=copy.deepcopy(model.state_dict());mask=None
    mapping={'FT':'FT','RL':'RL','GA':'GA','IU':'wfisher','BE':'boundary_expanding',
             'BS':'boundary_shrink','l1_sparse':'FT_l1','SalUn':'RL','SalUn_soft':'RL_proximal'}
    settings={'FT':(.01,10,.2),'RL':(.01,10,.2),'GA':(.0001,5,.2),
              'IU':(.01,1,10),'BE':(.00001,10,.2),'BS':(.00001,10,.2),
              'l1_sparse':(.01,10,.00001),'SalUn':(.013,10,.2),'SalUn_soft':(.013,10,.2),
              'MMU2T':(.001,5,2.)}
    if a.settings:
        chosen_settings=json.loads(a.settings.read_text())
        for name,v in chosen_settings.items():
            if name in settings:settings[name]=(v['lr'],v['epochs'],v['alpha'])
    else:chosen_settings={}
    grid={
      'FT':[(v,10,.2,.5) for v in [.001,.01,.1]],
      'RL':[(v,10,.2,.5) for v in [.001,.01,.1]],
      'GA':[(v,5,.2,.5) for v in [.00001,.0001,.001]],
      'IU':[(.01,1,v,.5) for v in [1,10,20]],
      'BE':[(v,10,.2,.5) for v in [.000001,.00001,.0001]],
      'BS':[(v,10,.2,.5) for v in [.000001,.00001,.0001]],
      'l1_sparse':[(lr,10,v,.5) for lr in [.001,.01,.1] for v in [.000001,.00001,.0001]],
      'SalUn':[(lr,10,.2,d) for lr in [.0005,.005,.013,.05] for d in [.1,.5,.9]],
      'SalUn_soft':[(lr,10,.2,d) for lr in [.0005,.005,.013,.05] for d in [.1,.5,.9]],
      'MMU':[(lr,5,1,.5) for lr in [.0001,.0005,.001]],
      'MMU2T':[(lr,5,m,.5) for lr in [.005,.01,.02] for m in [.05,.1,.2]]}
    jobs=[]
    for name in a.methods.split(','):
        if a.tune and name!='retrain':
            jobs.extend((name,f'{name}_candidate{i}',v) for i,v in enumerate(grid[name]))
        else:
            default=(*settings[name],chosen_settings.get(name,{}).get('density',.5)) if name in settings else (chosen_settings.get(name,{}).get('lr',.0005),5,1,.5)
            jobs.append((name,name,default))
    if a.tune:
        out=out/'tuning';out.mkdir(exist_ok=True)
    masks_cache={}
    for name,key,hp in jobs:
        result=out/f'{key}.json'
        if result.exists():continue
        utils.setup_seed(a.seed);student=copy.deepcopy(model);student.load_state_dict(source_state)
        start=time.perf_counter()
        if name=='retrain':
            hp=(.1,182,0,1)
            ck=(out.parent if a.tune else out)/'retrain.pt'
            if ck.exists():
                obj=torch.load(ck,weights_only=False);student.load_state_dict(obj['state_dict']);seconds=obj['seconds']
            else:
                student.load_state_dict(initial);seconds=fit(student,retain,va.dataset,args,ck)
        elif name in ('MMU','MMU2T'):
            spec=importlib.util.spec_from_file_location('mmu_method',REPO/'mmu/MMU/mmu.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            method=mod.MSCRUB(student,epochs=5,msteps=2,lr=hp[0],diagnose=False,
                              bad_margin=hp[2] if name=='MMU2T' else None)
            method.unlearn(data['forget'],data['retain']);seconds=time.perf_counter()-start
        else:
            args.unlearn=mapping[name];args.unlearn_lr,args.unlearn_epochs,args.alpha,args.mask_ratio=hp
            args.save_dir=str(out/key);Path(args.save_dir).mkdir(exist_ok=True)
            if name=='SalUn':
                if hp[3] not in masks_cache:masks_cache[hp[3]]=mask_for(model,forget,hp[3])
                mask=masks_cache[hp[3]]
            unlearn.get_unlearn_method(args.unlearn)(data,student,nn.CrossEntropyLoss(),args,mask if name=='SalUn' else None)
            if name=='BE':
                for module in student.modules():
                    for child_name,child in list(module.named_children()):
                        if isinstance(child,nn.Linear) and child.out_features==11:
                            head=nn.Linear(child.in_features,10,bias=child.bias is not None).cuda()
                            with torch.no_grad():
                                head.weight.copy_(child.weight[:10])
                                if child.bias is not None:head.bias.copy_(child.bias[:10])
                            setattr(module,child_name,head)
            seconds=time.perf_counter()-start
        if not a.tune and name in ('MMU','MMU2T','SalUn'):torch.save({'state_dict':student.state_dict(),'seconds':seconds},out/f'{key}.pt')
        metrics=evaluate(student,retain,forget,va.dataset if a.tune else te.dataset);metrics['RTE_min']=seconds/60
        metrics.update(seed=a.seed,ratio=a.ratio,method=name,protocol='paper-aligned-tuned-v1',evaluation_split='validation' if a.tune else 'test',
                       hyperparameters=dict(lr=hp[0],epochs=hp[1],alpha=hp[2],density=hp[3]))
        dump(result,metrics);print('RESULT',metrics,flush=True)
    if a.tune:
        ref=json.loads((out/'retrain.json').read_text())
        selected={}
        for name in a.methods.split(','):
            if name=='retrain':continue
            candidates=[json.loads(p.read_text()) for p in out.glob(f'{name}_candidate*.json')]
            for r in candidates:
                r['selection_gap']=sum(abs(r[k]-ref[k]) for k in ['UA','RA','TA','MIA'])/4
            best=min(candidates,key=lambda r:r['selection_gap'])
            selected[name]={**best['hyperparameters'],'validation_gap':best['selection_gap']}
        prior=json.loads((out/'selected.json').read_text()) if (out/'selected.json').exists() else {}
        prior.update(selected);dump(out/'selected.json',prior)

if __name__=='__main__':main()
