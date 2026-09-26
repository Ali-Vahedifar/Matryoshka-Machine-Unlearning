import os
import copy,importlib.util,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import classification as C
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostic_one_seed';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('mmu_impl',ROOT.parents[1]/'mmu/MMU/mmu.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
SEED=2

def norm(params):
    v=[p.grad.detach().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(v).sum().sqrt()) if v else 0.


def snapshot(method,teacher,loaders,stage):
    was=method.model.training;method.model.eval();result={'stage':stage,'sets':{}}
    with torch.no_grad(),torch.random.fork_rng(devices=[0]):
        for name,dl in loaders.items():
            counts={m:0 for m in method.granularities};kl={m:0. for m in counts};n=0
            for x,y in dl:
                x,y=x.cuda(),y.cuda();n+=len(y)
                student=method._student_logits(x)
                target=M.nested_logits(teacher,M.linear_head(teacher),x,method.granularities)
                for m,s,t in zip(method.granularities,student,target):
                    counts[m]+=int((s.argmax(1)==y).sum())
                    kl[m]+=float(M.kd_kl(s,t,method.temperature))*len(y)
            result['sets'][name]={str(m):{'accuracy':100*counts[m]/n,'teacher_kl':kl[m]/n} for m in counts}
        result['parameter_distance_l2']=float(torch.stack([(p-q).square().sum() for p,q in zip(method.model.parameters(),teacher.parameters())]).sum().sqrt())
    method.model.train(was)
    return result


def initial_gradient(method,teacher,x,mode):
    probe=copy.deepcopy(method);probe.model.train(mode=='train')
    pars=list(probe.model.parameters())+[p for h in (probe.heads or {}).values() for p in h.parameters()]
    with torch.no_grad():targets=M.nested_logits(teacher,M.linear_head(teacher),x,probe.granularities)
    students=probe._student_logits(x)
    loss=-sum(M.kd_kl(s,t,probe.temperature) for s,t in zip(students,targets))
    loss.backward()
    return {'student_mode':mode,'teacher_mode':'eval','loss':float(loss.detach()),
            'gradient_l2':norm(pars),
            'max_logit_difference':max(float((s-t).detach().abs().max()) for s,t in zip(students,targets))}


def run(source,forget,retain,probes,name,full):
    student=copy.deepcopy(source)
    kwargs={'granularities':[M.linear_head(student).in_features]} if full else {}
    method=M.MSCRUB(student,epochs=5,msteps=2,lr=.001,diagnose=False,**kwargs)
    teacher=M.clone_frozen(student);teacher_head=M.linear_head(teacher)
    method.heads=M.matryoshka_heads(method.head,method.granularities,'cuda')
    pars=list(student.parameters())+[p for h in method.heads.values() for p in h.parameters()]
    x,_=next(iter(probes['forget']));x=x.cuda()
    initial=[initial_gradient(method,teacher,x,mode) for mode in ['eval','train']]
    C.utils.setup_seed(SEED)
    opt=torch.optim.SGD(pars,lr=method.lr,momentum=method.momentum,weight_decay=method.weight_decay)
    a,b=method._scales();records=[snapshot(method,teacher,probes,'source')];updates=[]
    for epoch in range(method.epochs):
        for phase,dl in [('forget',forget),('retain',retain)]:
            if phase=='forget' and epoch>=method.msteps:continue
            student.train();loss_sum=gn_sum=0.;batches=0
            for x,y in dl:
                x,y=x.cuda(),y.cuda()
                with torch.no_grad():targets=M.nested_logits(teacher,teacher_head,x,method.granularities)
                students=method._student_logits(x);opt.zero_grad(set_to_none=True)
                if phase=='forget':loss=-sum(w*M.kd_kl(s,t,method.temperature) for w,s,t in zip(a,students,targets))
                else:loss=sum(w*(method.gamma*F.cross_entropy(s,y)+method.alpha*M.kd_kl(s,t,method.temperature)) for w,s,t in zip(b,students,targets))
                loss.backward();gn=norm(pars);opt.step()
                loss_sum+=float(loss.detach());gn_sum+=gn;batches+=1
            updates.append({'epoch':epoch,'phase':phase,'batches':batches,'mean_loss':loss_sum/batches,'mean_gradient_l2':gn_sum/batches})
            record=snapshot(method,teacher,probes,f'epoch{epoch}_{phase}');records.append(record)
            print(name,record['stage'],record['sets']['forget'][str(method.head.in_features)],flush=True)
    assert all(p.grad is None for p in teacher.parameters())
    result={'name':name,'seed':SEED,'widths':list(method.granularities),'initial_gradient':initial,'passes':updates,'probes':records}
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    sys.argv=['diagnostic'];args=C.arg_parser.parse_args();args.data=os.environ.get('MMU_DATA','data');args.seed=args.train_seed=SEED
    original=C.utils.cifar10_dataloaders
    def same_split(*v,**kw):kw.setdefault('seed',SEED);return original(*v,**kw)
    C.utils.cifar10_dataloaders=same_split
    model,tr,va,te,_=C.utils.setup_model_dataset(args);model.cuda()
    source=ROOT/f'classification/seed{SEED}/source.pt'
    model.load_state_dict(torch.load(source,weights_only=False)['state_dict']);model.eval()
    split=np.load(ROOT/f'classification/seed{SEED}/forget10/split.npz')
    def take(ds,idx):
        d=copy.deepcopy(ds);d.data=d.data[idx];d.targets=np.asarray(d.targets)[idx];return d
    forget=take(tr.dataset,split['forget']);retain=take(tr.dataset,split['retain'])
    forget=take(tr.dataset,np.sort(split['forget']))
    rng=np.random.default_rng(SEED)
    ri=rng.choice(len(retain),2048,replace=False);ti=rng.choice(len(te.dataset),2048,replace=False)
    probe_ds={'forget':copy.deepcopy(forget),'retain_probe':take(retain,ri),'test_probe':take(te.dataset,ti)}
    for ds in probe_ds.values():ds.transform=transforms.ToTensor()
    probes={k:C.loader(v) for k,v in probe_ds.items()}
    config={'seed':SEED,'forget_ratio':.1,'source':str(source),'epochs':5,'forget_epochs':2,'lr':.001,
            'temperature':4,'alpha':1,'gamma':1,'batch_size':256,'forget_samples':len(forget),
            'retain_training_samples':len(retain),'retain_probe_samples':2048,'test_probe_samples':2048,
            'controls':['fullwidth_scrub','nested_mmu'],
            'note':'Full-width control is the existing SCRUB-style MMU objective with only its full-width head; not an independent released SCRUB reproduction. No hyperparameter sweep.'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2));np.savez(OUT/'probe_indices.npz',retain=ri,test=ti)
    for name,full in [('fullwidth_scrub',True),('nested_mmu',False)]:
        run(model,C.loader(forget,True),C.loader(retain,True),probes,name,full)
    print('ONE-SEED DIAGNOSTIC COMPLETE',flush=True)

if __name__=='__main__':main()
