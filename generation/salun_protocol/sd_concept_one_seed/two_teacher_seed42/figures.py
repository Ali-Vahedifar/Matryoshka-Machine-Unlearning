import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT=Path(__file__).resolve().parent
ROWS=[('source','Source SD'),('mmu_2t_hi','MMU'),('random_mask','Random mask'),('salun_port','SalUn')]
SETS={0:('FORGET SET (removed): nude adult women, P1–P10','#c62828'),
      10:('FORGET SET (removed): nude adult men, P11–P20','#c62828'),
      20:('RETAIN SET (kept): clothed people, P21–P30','#1a7f37'),
      30:('RETAIN SET (kept): other objects, P31–P40','#1a7f37')}


def draw(starts, results):
    fig=plt.figure(figsize=(20,7.8*len(starts)))
    blocks=fig.add_gridspec(len(starts),1,left=.135,right=.995,top=.972,bottom=.012,hspace=.14)
    for block,start in enumerate(starts):
        cells=blocks[block].subgridspec(4,10,wspace=.025,hspace=.055)
        for i,(name,label) in enumerate(ROWS):
            for j in range(10):
                idx=start+j;ax=fig.add_subplot(cells[i,j])
                ax.imshow(Image.open(ROOT/'previews'/name/f'{idx:03}.png'))
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(results[name]['records'][idx]['detected'])
                    spine.set_color('#c62828');spine.set_linewidth(2)
                text,color=SETS[start]
                if i==0:ax.set_title(f'P{idx+1}',fontsize=17,fontweight='bold',color=color)
                if i==0 and j==0:ax.text(0,1.3,text,transform=ax.transAxes,fontsize=19,fontweight='bold',color=color)
                if j==0:ax.set_ylabel(label,fontsize=17,fontweight='bold',labelpad=14)
    return fig


def main():
    archive=ROOT/'archive_six_rows';archive.mkdir(exist_ok=True)
    for name in ['comparison.pdf','comparison_00.pdf','comparison_10.pdf','comparison_20.pdf','comparison_30.pdf']:
        p=ROOT/name
        if p.exists() and not (archive/name).exists():shutil.copyfile(p,archive/name)
    results=json.loads((ROOT/'results.json').read_text())
    with PdfPages(ROOT/'comparison.pdf') as combined:
        for name,starts in [('figure_1',[0]),('figures_2_3_4',[10,20,30])]:
            fig=draw(starts,results)
            fig.savefig(ROOT/f'{name}.pdf',dpi=200,bbox_inches='tight',pad_inches=.04)
            fig.savefig(ROOT/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.04)
            combined.savefig(fig,dpi=200,bbox_inches='tight',pad_inches=.04)
            plt.close(fig)
    for start in [0,10,20,30]:
        fig=draw([start],results)
        fig.savefig(ROOT/f'comparison_{start:02}.pdf',dpi=200,bbox_inches='tight',pad_inches=.04)
        plt.close(fig)
    (ROOT/'figure_selection.json').write_text(json.dumps(dict(
        training_seed=42,rows=ROWS,mmu_variant='mmu_2t_hi',margin=.002,
        note='Selected higher-margin full-width variant; all variant metrics remain in report.md and results.json.',
        pages=[[0,9],[10,39]],detector_borders=True,presentation_censoring='unchanged'),indent=2))
    p=ROOT/'report.md';s=p.read_text()
    s=s.replace('[All four comparison pages](comparison.pdf)', '[Selected four-row comparison, two pages](comparison.pdf)')
    note='\nFigure exports: [Figure 1](figure_1.pdf) and [Figures 2–4 combined](figures_2_3_4.pdf). Rows are Source SD, MMU, Random mask, and SalUn. The displayed MMU is mmu_2t_hi (margin 0.002, seed 42); the complete six-row figures are preserved in archive_six_rows/. Figure labels omit titles and variant details; the full results above are unchanged.\n'
    if 'Figure exports:' not in s:s+=note
    p.write_text(s)
    print('FIGURES COMPLETE')


if __name__=='__main__':main()
