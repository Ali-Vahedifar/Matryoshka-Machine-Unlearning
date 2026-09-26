import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
methods=['retrain','FT','RL','GA','IU','BE','BS','l1_sparse','SalUn','SalUn_soft','MMU','MMU2T']
lines=['# Classification: paper-aligned SalUn comparison','',
       'Pilot seed 1 selects settings using validation metrics. Reported test trials are seeds 2–11, each with its own source model and paired retraining reference. '
       'Metrics are D_f (forget accuracy = 100 − UA), D_r (retain accuracy), Test accuracy and confidence-based SVC MIA; smaller absolute gaps to Retrain are preferred. '
       'This uses an explicit coarse search within the published ranges, not an exact reproduction of unpublished grid choices.','']
for ratio in [10,50]:
    data={m:[] for m in methods}
    for seed in range(2,12):
        for m in methods:
            p=ROOT/f'classification/seed{seed}/forget{ratio}/{m}.json'
            if p.exists():data[m].append(json.loads(p.read_text()))
    refs={r['seed']:r for r in data['retrain']}
    lines += [f'## {ratio}% random forgetting','',
              '| Method | Trials | D_f accuracy (gap ↓) | D_r accuracy (gap ↓) | Test accuracy (gap ↓) | MIA (gap ↓) | Avg. gap ↓ | RTE minutes ↓ |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,records in data.items():
        records=[r for r in records if r['seed'] in refs]
        if not records:continue
        cells=[];gaps=[]
        for k in ['UA','RA','TA','MIA']:
            v=[100-r[k] if k=='UA' else r[k] for r in records];mean=statistics.mean(v);sd=statistics.stdev(v) if len(v)>1 else 0
            gap=abs(mean-statistics.mean(100-refs[r['seed']][k] if k=='UA' else refs[r['seed']][k] for r in records));gaps.append(gap)
            cells.append(f'{mean:.2f} ± {sd:.2f} ({gap:.2f})')
        avg=statistics.mean(gaps);rte=statistics.mean(r['RTE_min'] for r in records)
        lines.append(f'| {m} | {len(records)} | '+' | '.join(cells)+f' | {avg:.2f} | {rte:.2f} |')
    lines += ['']
(ROOT/'classification_summary.md').write_text('\n'.join(lines))
print(ROOT/'classification_summary.md')
