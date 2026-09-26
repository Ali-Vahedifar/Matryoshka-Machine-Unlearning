import os
import csv,json,sys,logging
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'diffusion_mmu'))
from evaluate import judge,classify,CLASSES
logging.getLogger('fontTools').setLevel(logging.WARNING)
OUT=ROOT/'scientific_figures'; WORK=Path(os.environ.get('MMU_WORK','work')+'/mmu_cifar_all_classes')
METHODS=['MMU','Random mask','SalUn']; COLORS=['#0072B2','#009E73','#D55E00']; SHORT=['Air','Auto','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':250})

def heat(ax,acc):
    im=ax.imshow(acc,vmin=80,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(3),METHODS);ax.set_yticks(range(10),CLASSES)
    ax.set_ylabel('Class removed');ax.set_title('Retained-condition accuracy ↑',loc='left',fontweight='bold',pad=12)
    for c in range(10):
        for j in range(3):ax.text(j,c,f'{acc[c,j]:.1f}',ha='center',va='center',color='white' if acc[c,j]>92 else 'black',fontsize=10)
    return im

def transitions(ax,mat,name):
    im=ax.imshow(mat,vmin=0,vmax=100,cmap='Blues')
    ax.set_xticks(range(10),SHORT,rotation=60,ha='right',fontsize=8);ax.set_yticks(range(10),SHORT,fontsize=8)
    ax.set_title(name+' (%)',fontweight='bold',loc='left');ax.set_xlabel('Predicted class');ax.set_ylabel('Forgotten condition')
    for c in range(10):
        for k in range(10):
            if mat[c,k]>0:ax.text(k,c,f'{mat[c,k]:.0f}',ha='center',va='center',fontsize=7,color='white' if mat[c,k]>50 else 'black')
    return im

def scatter(ax,changes,labels=True):
    for j,name in enumerate(METHODS):
        v=np.array(changes[name]);ax.scatter(v[:,0],v[:,1],s=65,c=COLORS[j],marker=['o','s','^'][j],label=name,alpha=.85,edgecolors='white',linewidth=.7)
        if labels:
            for c,(x,y) in enumerate(v):ax.annotate(str(c),(x,y),xytext=(4,4),textcoords='offset points',fontsize=8,color=COLORS[j])
    ax.set_xlabel('Forgotten-condition change from source (MSE) →')
    ax.set_ylabel('Retained-condition change from source (MSE) ↓')
    ax.set_title('Selective change in generated images',loc='left',fontweight='bold',pad=12)
    ax.grid(alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False)
    ax.text(.98,.02,'Desired direction: lower right',transform=ax.transAxes,ha='right',fontsize=9,color='#555555')

def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4)
    cached=torch.load(ROOT/'expanded/new_samples.pt',map_location='cpu',weights_only=False)['images']
    source=cached['Source'].reshape(10,16,3,32,32)
    metrics=json.loads((ROOT/'all_classes_seed42/metrics.json').read_text())
    acc=np.array([[100*metrics[str(c)][m]['retain_correct']/144 for m in METHODS] for c in range(10)])
    clf=judge();preds={'Source':classify(source.flatten(0,1).cuda(),clf).cpu().reshape(10,16)}
    mats={'Source':np.stack([np.bincount(p.numpy(),minlength=10)*100/16 for p in preds['Source']])}
    changes={m:[] for m in METHODS};records=[];sample_predictions={}
    for j,m in enumerate(METHODS):
        rows=[]
        for c in range(10):
            stem=['two_teacher' if c==0 else 'mmu','random','salun'][j]
            v=torch.load(WORK/f'class_{c}'/f'{stem}_samples.pt',map_location='cpu',weights_only=True)
            assert v.shape==source.shape and torch.isfinite(v).all()
            pred=classify(v.flatten(0,1).cuda(),clf).cpu().reshape(10,16)
            correct=pred==torch.arange(10)[:,None]; keep=torch.arange(10)!=c
            assert int(correct[c].sum())==metrics[str(c)][m]['forget_correct']
            assert int(correct[keep].sum())==metrics[str(c)][m]['retain_correct']
            rows.append(np.bincount(pred[c].numpy(),minlength=10)*100/16)
            error=((v-source)/2).square().mean((2,3,4))
            f=float(error[c].mean());r=float(error[keep].mean());changes[m].append([f,r])
            records.append(dict(method=m,forgotten_class=CLASSES[c],class_id=c,forget_mse=f,retain_mse=r,retain_accuracy=acc[c,j]))
            sample_predictions[f'{m}/class_{c}']=pred.tolist()
        mats[m]=np.stack(rows)
    (OUT/'predictions.json').write_text(json.dumps(sample_predictions,indent=2))
    (OUT/'plot_data.json').write_text(json.dumps(dict(class_names=CLASSES,methods=METHODS,retained_accuracy=acc.tolist(),transition_percent={m:v.tolist() for m,v in mats.items()},paired_mse=changes),indent=2))
    with (OUT/'paired_changes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    figs=[]
    fig,ax=plt.subplots(figsize=(5.8,6.5),layout='constrained');im=heat(ax,acc);fig.colorbar(im,ax=ax,label='Accuracy (%)');save(fig,'retained_accuracy_heatmap');figs.append(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4.5),layout='constrained')
    for ax,(name,mat) in zip(axes,mats.items()):im=transitions(ax,mat,name)
    fig.colorbar(im,ax=axes,label='Samples (%)',shrink=.7);save(fig,'class_transition_matrices');figs.append(fig)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');scatter(ax,changes)
    fig.text(.5,-.025,'Class IDs: '+', '.join(f'{i} {s}' for i,s in enumerate(SHORT)),ha='center',fontsize=8)
    save(fig,'preservation_forgetting_scatter');figs.append(fig)
    fig=plt.figure(figsize=(17,8),layout='constrained');outer=fig.add_gridspec(1,3,width_ratios=[.8,1.3,1.15])
    ax=fig.add_subplot(outer[0]);im=heat(ax,acc);ax.set_title('(a) Retained accuracy ↑',loc='left',fontweight='bold');fig.colorbar(im,ax=ax,orientation='horizontal',label='Accuracy (%)',shrink=.8)
    sub=outer[1].subgridspec(2,2)
    for i,(name,mat) in enumerate(mats.items()):
        ax=fig.add_subplot(sub[i//2,i%2]);transitions(ax,mat,('(b) ' if i==0 else '')+name)
    ax=fig.add_subplot(outer[2]);scatter(ax,changes,False);ax.set_title('(c) Selective image change',loc='left',fontweight='bold')
    save(fig,'comparison_analysis');figs.append(fig)
    histories={name:json.loads((ROOT/'two_teacher_seed42'/f'{key}_training.json').read_text())['history'] for name,key in [('MMU','two_teacher'),('Attraction only','frozen_redirect')]}
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    specs=[('attraction','Attraction to alternative-condition teacher'),('distance','Distance from forgotten-condition teacher'),('hinge','Margin repulsion'),('retain','Retained denoising + preservation (weighted)')]
    for ax,(key,title) in zip(axes.flat,specs):
        for name,hist in histories.items():
            vals=[3.072*(r['denoise']+r['preservation']) if key=='retain' else r[key] for r in hist]
            ax.plot([r['step']+1 for r in hist],vals,label=name,color=COLORS[0] if name=='MMU' else '#CC79A7',linestyle='-' if name=='MMU' else '--',marker='o' if name=='MMU' else 's',markersize=4)
        if key=='distance':ax.axhline(.002,color='#777777',ls=':',label='Margin δ = 0.002')
        ax.set_title(title,loc='left',fontsize=11,fontweight='bold');ax.set_xlabel('Optimizer update');ax.set_ylabel('Loss / mean squared discrepancy');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    save(fig,'mmu_mechanism_control');figs.append(fig)
    with PdfPages(OUT/'all_scientific_figures.pdf') as pdf:
        for fig in figs:pdf.savefig(fig,bbox_inches='tight')
    for fig in figs:plt.close(fig)
    lines=['# Scientific figures: one-seed CIFAR-10 analysis','',
      'Generated entirely from saved samples, classifier evaluations, and training logs. No additional training or image generation. PDF, SVG, and PNG versions are provided for each figure.','',
      '## Figures','',
      '- [Retained-accuracy heatmap](retained_accuracy_heatmap.pdf): each cell evaluates 144 retained-condition samples for one deleted class. Fixed color scale 80–100%; values are printed.',
      '- [Class-transition matrices](class_transition_matrices.pdf): each row evaluates 16 samples under the corresponding forgotten condition from a separately unlearned model. Source is the unchanged reference. All matrices share 0–100%; displayed percentages are rounded to the nearest integer, with exact values saved in plot_data.json.',
      '- [Preservation–forgetting scatter](preservation_forgetting_scatter.pdf): each point is one deletion task. Coordinates are mean paired RGB MSE relative to source, after scaling to [0,1]. Horizontal uses 16 forgotten-condition samples; vertical uses 144 retained-condition samples. Larger horizontal change and smaller vertical change indicate selectivity, but neither proves semantic forgetting or image quality.',
      '- [Three-panel comparison](comparison_analysis.pdf): compact combination of the preceding analyses.',
      '- [MMU mechanism and attraction-only control](mmu_mechanism_control.pdf): airplane only, training seed 42, 1,000 updates. Full-width MMU and an otherwise matched attraction-only control. Lines connect sparse recorded losses; they are not full trajectories. The zero control repulsion curve is disabled by construction, not evidence that the margin was satisfied. Loss magnitude is not gradient magnitude.',
      '- [All figures in one PDF](all_scientific_figures.pdf).','',
      '## Interpretation and limitations','',
      'All three unlearning methods have zero forgotten-class detections across 160 evaluated samples. Average retained-condition accuracy is MMU 97.92%, Random mask 98.75%, SalUn 94.03%. These are local implementations, not an official SalUn reproduction. Ten deletion classes are tasks, not independent training seeds; no confidence interval or superiority test is claimed.',
      'All methods deliberately redirect forgotten class c toward class (c+1) modulo 10. Transition matrices show whether generation follows that training target; they do not independently demonstrate data removal. Source samples and initial noise are reused across tasks and methods.',
      'The existing airplane control and MMU both produced 0/32 forgotten-class detections and 285/288 retained correct predictions in the earlier two-noise-seed evaluation. Those equal endpoints do not establish a benefit from repulsion. Mechanism curves use that original airplane training log, while the ten-class plots use the single matched sampling seed 10043 (16 samples per condition).','',
      'Raw plotted values: [JSON](plot_data.json), [paired errors CSV](paired_changes.csv), [classifier predictions](predictions.json). Reproduce with `python salun_protocol/generation/scientific_figures.py` from generation/.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print('SCIENTIFIC FIGURES COMPLETE',flush=True)

if __name__=='__main__':main()
