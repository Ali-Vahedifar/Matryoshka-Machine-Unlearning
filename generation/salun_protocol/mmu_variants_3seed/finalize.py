import os
import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import run as S
R=S.R;W=S.W

def main():
 while True:
  status=json.loads((R/'status.json').read_text())
  if status['status']=='failed':raise RuntimeError(status['traceback'])
  if status['status']=='complete':break
  time.sleep(20)
 torch.set_num_threads(4);all_rows=json.loads((R/'results.json').read_text())
 for mode in ['class','instance','subclass']:
  for seed in [42,43,44]:
   key=f'{mode}/seed{seed}';b,old,out=S.setup(mode,seed)
   ds=copy.deepcopy(b.loaders['retain_full'].dataset);base=ds
   while hasattr(base,'dataset'):base=base.dataset
   clean=b.loaders['forget_train_eval'].dataset
   while hasattr(clean,'dataset'):clean=clean.dataset
   base.transform=clean.transform
   retain=DataLoader(ds,batch_size=256,shuffle=False,num_workers=0)
   forget=DataLoader(b.loaders['forget_train_eval'].dataset,batch_size=256,shuffle=False,num_workers=0)
   for id,row in all_rows[key].items():
    if row.get('status')=='failed':continue
    if 'Df_accuracy' not in row:
     model=b._fresh_source()
     if id=='Retrain':model.load_state_dict(b.reference_state)
     elif id!='Source':model.load_state_dict(torch.load(out/f'{id}.pt',weights_only=False)['model'])
     model.eval();row['Df_accuracy']=100*S.B.evaluate_task(model,forget,b.device);row['Dr_accuracy']=100*S.B.evaluate_task(model,retain,b.device)
     row['Test_accuracy']=row['metrics']['test_acc'];row['MIA_attack_accuracy']=row['metrics']['mia'];row['Df_n']=len(forget.dataset);row['Dr_n']=len(retain.dataset)
     S.dump(out/f'{id}.json',row);S.dump(R/'results.json',all_rows)
   ref=all_rows[key]['Retrain']
   for row in all_rows[key].values():
    if row.get('status')=='failed':continue
    row['avg_gap']=float(np.mean([abs(row[k]-ref[k]) for k in ['Df_accuracy','Dr_accuracy','Test_accuracy','MIA_attack_accuracy']]))
   S.dump(R/'results.json',all_rows);print('CLEAN METRICS',key,flush=True)
 catalogue=json.loads((R/'variants.json').read_text())
 lines=['# MMU variants: three-seed CIFAR-10 classification','', 'Status: complete. ResNet-18; seeds 42, 43, 44. Matched cached source and retrain checkpoints passed protocol and content checks; all unlearning variants and SalUn were rerun. No new hyperparameter selection on test results.','',
 'Class deletion removes airplane from the 10-way task. Instance deletion removes 4,500 class-balanced training examples (10% of the training split). Subclass deletion uses the existing custom five-superclass CIFAR-10 grouping: airplane/ship, automobile/truck, bird/frog, cat/dog, deer/horse. It removes airplane while its sibling ship remains. CIFAR-10 has no official subclass hierarchy.','',
 'D_f and D_r below are accuracies on the complete forgotten and retained training subsets with augmentation disabled. Test accuracy uses the official test set (5-way labels for subclass deletion). MIA is the existing independently calibrated confidence-attack accuracy, not the SalUn-protocol efficacy metric. Gap averages absolute deviations from paired Retrain across D_f, D_r, Test and MIA. Raw metrics target retrain; lower raw D_f is not universally better for instance/subclass deletion.','',
 'The original validation-selected MMU lr/epochs/forget-pass count is shared within each cell by source-role variants. The margin is fixed at 0.05 in temperature-4 KL units; it is not the diffusion MSE margin. Retain-FT adaptations use five joint epochs, lr .001, temperature 1, and margin .002 where present. All post-hoc students use the same 10% retained-data subset. Their retain-FT attraction teacher also uses that subset for five epochs, and its cost is included. SalUn uses its existing validation-selected settings; all per-cell settings are saved. Thus schedules are controlled within families, not identical across every method.','',
 '## Variant key','', '| ID | Name | Teacher target | Forgotten attraction |','|---|---|---|---|']
 for v in catalogue:lines.append(f'| {v["id"]} | {v["name"]} | {v["teacher"]} | {"yes" if v["attraction_forget"] else "no"} |')
 lines+=['','MRL = separate prefix heads; MRL-E = one sliced shared head. Sum = unnormalised width losses. Weighted mean = normalised original MMU weights (uniform forget weights and inverse-width retained weights). Source-role variants preserve the frozen source on D_r and repel it on D_f; this is one frozen network with two roles. Retain-FT adaptations use two distinct frozen networks and should not be confused with the diffusion method’s two conditioned predictions. No variant uses a moving full-width student as its teacher.','']
 short={'Source':'source reference','Retrain':'retain-only retraining','SalUn':'local baseline'}
 for v in catalogue:
  prefix='nested '+('separate' if v['head']=='independent' else 'shared') if v['nest'] else 'full width'
  if v['family']=='retain_ft_adapter':detail='FT + source / attraction'+(' + margin' if v['margin'] is not None else ' only')
  else:detail=('full source' if v['teacher']=='full source' else ('sum' if v['weight']=='mrl' else 'weighted mean'))+' / '+('unbounded' if v['margin'] is None else 'margin')
  short[v['id']]=prefix+' / '+detail
 ids=['Source','Retrain','SalUn']+[v['id'] for v in catalogue]
 for mode in ['class','instance','subclass']:
  lines+=['## '+mode.capitalize()+' deletion','', '| Variant | D_f | D_r | Test | MIA | Gap ↓ | Seconds ↓ | Complete seeds |','|---|---:|---:|---:|---:|---:|---:|---:|']
  for id in ids:
   rows=[all_rows[f'{mode}/seed{s}'].get(id,{}) for s in [42,43,44]];ok=[v for v in rows if 'Df_accuracy' in v]
   if len(ok)!=3:
    lines.append(f'| {id} | — | — | — | — | — | — | {len(ok)}/3 |');continue
   vals=[[v[k] for v in ok] for k in ['Df_accuracy','Dr_accuracy','Test_accuracy','MIA_attack_accuracy','avg_gap']]
   vals.append([v['seconds']+v.get('teacher_seconds',0) for v in ok])
   cells=[f'{np.mean(x):.2f} ± {np.std(x,ddof=1):.2f}' for x in vals]
   if id=='Source':cells[-1]='—'
   lines.append('| '+id+': '+short[id]+' | '+' | '.join(cells)+' | 3/3 |')
 lines+=['','Times include teacher construction for retain-FT variants. Source training is excluded, while retrain time is the cached full-retraining measurement. Three training seeds provide a variability estimate, not proof of superiority. Class and subclass each test one fixed deletion target. Underlying per-seed results, checkpoints and auxiliary heads are preserved under $MMU_WORK/mmu_variants_3seed.']
 (R/'report.md').write_text('\n'.join(lines))
 S.dump(R/'final_status.json',dict(status='complete',cells=9,variants=15));print('THREE-SEED REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
