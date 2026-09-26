import json,logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
logging.getLogger('fontTools').setLevel(logging.WARNING)
R=Path(__file__).resolve().parent;O=R/'redesigned_figures';O.mkdir(exist_ok=True)
d=json.loads((R/'scientific_figures/plot_data.json').read_text());m=json.loads((R/'all_classes_seed42/metrics.json').read_text())
methods=['MMU','Random mask','SalUn'];colors=['#1765D1','#139888','#E66A33'];markers=['o','s','^'];classes=d['class_names']
loss=np.array([[100*(m[str(c)]['Source']['retain_correct']-m[str(c)][name]['retain_correct'])/144 for name in methods] for c in range(10)])
target=np.array([[d['transition_percent'][name][c][(c+1)%10] for name in methods] for c in range(10)])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#CBD1D8','text.color':'#172333','axes.labelcolor':'#172333','xtick.color':'#536071','ytick.color':'#536071','pdf.fonttype':42,'svg.fonttype':'none'})
figures=[]
def title(fig,head,sub):
 fig.text(.065,.96,head,fontsize=23,weight='bold',va='top');fig.text(.065,.90,sub,fontsize=11,color='#536071',va='top')
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(O/f'{name}.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
 figures.append(fig)

fig,ax=plt.subplots(figsize=(11,8));fig.subplots_adjust(left=.18,right=.95,top=.78,bottom=.14)
title(fig,'What survives after a class is removed?','Retained-condition accuracy loss relative to the source • ten separate deletion tasks')
order=np.argsort(-loss[:,2],kind='stable')
for row,c in enumerate(order):
 ax.axhspan(row-.43,row+.43,color='#F4F6F9' if row%2==0 else 'white',zorder=0)
 ax.plot([loss[c].min(),loss[c].max()],[row,row],color='#CBD1D8',lw=3,zorder=1)
 for j,name in enumerate(methods):
  ax.scatter(loss[c,j],row+(j-1)*.14,c=colors[j],s=80,marker=markers[j],edgecolor='white',linewidth=.8,zorder=3,label=name if row==0 else None)
ax.set_yticks(range(10),[classes[c].capitalize() for c in order]);ax.invert_yaxis();ax.tick_params(axis='y',length=0)
ax.set_xlim(-.4,14);ax.set_xticks(range(0,15,2));ax.set_xlabel('Retained accuracy lost (percentage points)  ↓',labelpad=12)
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False)
fig.text(.18,.825,'MEAN LOSS',fontsize=9,weight='bold',color='#536071')
for j in range(3):fig.text(.34+j*.205,.825,f'{methods[j]}  {loss[:,j].mean():.2f} pp',color=colors[j],fontsize=12,weight='bold')
fig.text(.18,.04,'Seed 42 · 144 retained samples per task · local implementations · lower is better',fontsize=9,color='#536071')
save(fig,'01_collateral_damage')

fig,axs=plt.subplots(1,3,figsize=(15,7),sharex=True,sharey=True);fig.subplots_adjust(left=.075,right=.97,top=.73,bottom=.20,wspace=.10)
title(fig,'Redirection has a preservation cost','Same replacement target, different collateral damage • each point is one deleted class')
for j,ax in enumerate(axs):
 ax.set_facecolor('#F7F9FC');ax.set_title(methods[j],color=colors[j],fontsize=17,weight='bold',pad=16)
 ax.axhline(100,color='#CBD1D8',ls=':',lw=1)
 groups={}
 for c in range(10):groups.setdefault((float(loss[c,j]),float(target[c,j])),[]).append(classes[c])
 top_label=0
 for idx,((x,y),names) in enumerate(sorted(groups.items())):
  ax.scatter(x,y,c=colors[j],marker=markers[j],s=85,zorder=3,edgecolors='white')
  text='\n'.join(names)
  if y==100:
   ax.annotate(', '.join(names),(x,y),xytext=(6.0,100-top_label*3.5),textcoords='data',fontsize=8,ha='left',va='center',color='#344255',arrowprops=dict(arrowstyle='-',color='#BBC5D1',lw=.7),bbox=dict(facecolor='#F7F9FC',edgecolor='none',pad=1.5))
   top_label+=1
  else:
   ax.annotate(text,(x,y),xytext=(6,-6),textcoords='offset points',fontsize=8,ha='left',va='top',color='#344255')
 ax.set_xlim(-.6,15);ax.set_ylim(76,104);ax.set_xticks([0,5,10,15]);ax.grid(alpha=.16);ax.set_axisbelow(True)
 ax.set_xlabel('Retained accuracy loss (pp) ↓')
 ax.text(.05,.035,f'Mean replacement success: {target[:,j].mean():.1f}%',transform=ax.transAxes,fontsize=9,color=colors[j],weight='bold')
axs[0].set_ylabel('Replacement-class predictions (%) ↑',labelpad=12)
fig.text(.075,.09,'Desired corner: upper left. Shared coordinates are grouped under one marker; no jitter or omitted tasks.',fontsize=10,color='#536071')
fig.text(.075,.05,'Target = next CIFAR-10 class (cyclic). 16 forgotten / 144 retained samples per task; seed 42. Redirection is not proof of data removal.',fontsize=9,color='#536071')
save(fig,'02_redirection_vs_disruption')

h={name:json.loads((R/'two_teacher_seed42'/f'{key}_training.json').read_text())['history'] for name,key in [('MMU','two_teacher'),('Attraction only','frozen_redirect')]}
e=json.loads((R/'two_teacher_seed42/metrics.json').read_text())
fig=plt.figure(figsize=(13,7));gs=fig.add_gridspec(1,2,left=.075,right=.97,top=.75,bottom=.20,width_ratios=[1.12,1],wspace=.25)
title(fig,'More separation. Same measured endpoints.','Airplane ablation: MMU versus an otherwise matched attraction-only control')
ax=fig.add_subplot(gs[0]);cc=['#1765D1','#9670B7']
for (name,history),color in zip(h.items(),cc):ax.plot([r['step']+1 for r in history],[1000*r['distance'] for r in history],color=color,label=name,lw=2.6,marker='o',ms=4)
ax.axhline(2,ls=':',color='#7F8998');ax.text(990,2.08,'MMU margin',ha='right',fontsize=9,color='#536071')
ax.set_xlabel('Optimizer update');ax.set_ylabel('Forgotten-teacher discrepancy (MSE × 10³)');ax.set_ylim(0,2.5);ax.grid(alpha=.15);ax.legend(frameon=False,loc='lower right');ax.set_title('Prediction-space separation',loc='left',fontsize=13,weight='bold',pad=15)
ax=fig.add_subplot(gs[1]);ax.axis('off');ax.set_title('Final generation behavior',loc='left',fontsize=13,weight='bold',pad=15)
for j,(name,key) in enumerate([('MMU','two_teacher'),('Attraction only','frozen_redirect')]):
 x=.37+j*.42;r=e[key]
 ax.text(x,.89,name,ha='center',fontsize=12,weight='bold',color=cc[j])
 ax.text(x,.63,f'{r["forget_airplane_count"]}/{r["forget_n"]}',ha='center',fontsize=27,weight='bold',color=cc[j])
 ax.text(x,.28,f'{r["retain_correct"]}/{r["retain_n"]}',ha='center',fontsize=25,weight='bold',color=cc[j])
ax.text(.02,.76,'Forgotten-class detections',fontsize=10,color='#536071');ax.text(.02,.42,'Correct retained conditions',fontsize=10,color='#536071')
ax.axhline(.51,color='#DFE4EA',lw=1);ax.text(.5,.03,'Equal observed counts ≠ established equivalence',ha='center',fontsize=10,color='#536071')
fig.text(.075,.09,'The repulsion term changes teacher separation; this one-seed control does not show an endpoint advantage.',fontsize=11,color='#344255')
fig.text(.075,.05,'Airplane only · training seed 42 · two sampling seeds · sparse training logs joined by lines · MSE is not gradient magnitude',fontsize=9,color='#536071')
save(fig,'03_repulsion_control')
with PdfPages(O/'all_three_figures.pdf') as pdf:
 for fig in figures:pdf.savefig(fig,bbox_inches='tight')
for fig in figures:plt.close(fig)
(O/'data.json').write_text(json.dumps(dict(classes=classes,methods=methods,retained_loss_pp=loss.tolist(),replacement_success_percent=target.tolist(),control_histories=h,control_endpoints={k:e[k] for k in ['two_teacher','frozen_redirect']}),indent=2))
(O/'README.md').write_text('''# Redesigned behavioral analysis

Three descriptive figures from existing samples and logs. No new training or sample selection.

1. **Collateral damage:** retained accuracy loss from source, by deleted class. Rows sorted by SalUn loss; small vertical offsets distinguish ties, without changing x-values. Each method has 144 retained samples per task.
2. **Redirection versus disruption:** success means the classifier predicts the prescribed replacement class, not simply any non-forgotten class. Identical coordinates are grouped, with all class names shown. Each task has 16 forgotten samples. The desired direction is upper left; redirection is not proof of data removal.
3. **Repulsion control:** airplane only. Teacher discrepancy from sparse batch-loss logs alongside final generated-sample counts for MMU and the matched attraction-only control. Equal counts are not a statistical equivalence result. This control uses 32 forgotten and 288 retained samples, spanning two sampling seeds.

All training uses seed 42 and local implementations. Ten deletion classes are separate tasks, not independent training seeds. No confidence intervals or significance claims. Baseline and MMU source data remain unchanged. PDF and SVG are vector exports; PNG is provided for preview. Exact plotted data are in data.json.
''')
print('Finished:',O)
