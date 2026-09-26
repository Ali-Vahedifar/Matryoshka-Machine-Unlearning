import json
import subprocess
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parent
base = root/'results/full_redirect_20260907_c0_seed42'
train_results = json.loads((base/'results.json').read_text())
out = root/'results/full_redirect_20260907_comparison'
out.mkdir(exist_ok=True)
merged, grids = {}, {}
paths = {}
for name in train_results:
    run = root/f'results/full_redirect_20260907_cfg2_{name}_c0_seed42'
    r = json.loads((run/'results.json').read_text())[name]
    r['evaluation_load_time_s'] = r['unlearn_time_s']
    r['unlearn_time_s'] = train_results[name]['unlearn_time_s']
    merged[name] = r
    grids[name] = torch.load(run/'grids.pt', map_location='cpu', weights_only=True)[name]
    paths[name] = str(run)
config = json.loads((base/'config.json').read_text())
config['guidance'] = 2.0
config['evaluation_runs'] = paths
(out/'config.json').write_text(json.dumps(config,indent=2))
(out/'results.json').write_text(json.dumps(merged,indent=2))
torch.save(grids,out/'grids.pt')
subprocess.run([sys.executable,str(root/'summarize_run.py'),str(out)],check=True)
lines = ['# Revised MMU-gen: complete single-seed experiment', '',
         'CIFAR-10, forgotten class airplane (0), seed 42. Existing source and class-0 retrain checkpoints; '
         '3 unlearning epochs, 13,500 retained training images, 5,000 forgotten training images, '
         'batch 128, learning rate 1e-4. Redirection uses retained content and the forgotten label, '
         'not the forgotten images. All methods use the same learning rate in this run.', '',
         'Each evaluation generates 500 forgotten-condition and 5,000 retained-condition images '
         'with 100 DDIM steps. Guidance 1 and 2 use identical saved model weights and evaluation seeds.', '',
         '| Method | UA g=1 | FID g=1 | Retain acc g=1 | UA g=2 | FID g=2 | Retain acc g=2 | Training/load seconds |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for name,r in train_results.items():
    q=merged[name]
    lines.append(f"| {name} | {r['UA']:.1f} | {r['FID_retain']:.2f} | {r['retain_cond_acc']:.1f} | "
                 f"{q['UA']:.1f} | {q['FID_retain']:.2f} | {q['retain_cond_acc']:.1f} | {r['unlearn_time_s']:.1f} |")
lines += ['', 'Source and retrain times measure loading, not their original training cost. '
          'Model evaluation time is excluded from unlearning times.', '',
          'The SalUn row is a local adaptation with the released prediction-matching direction, '
          'not a validated reproduction of published results. NegGrad uses 1e-4 here; '
          'the older experiment used 5e-5, so its rows are not directly interchangeable.', '',
          'The retained-only control uses nested widths with skip truncation. '
          'No parameter sweep or checkpoint selection was performed. '
          'One seed and one forgotten class do not establish general superiority. '
          'UA measures classifier rejection of the forgotten class, not removal of training information. '
          'Retained FID does not assess image quality under the forgotten condition.', '',
          f'[Guidance-1 report and grids]({base}/summary.md)', '',
          '[Guidance-2 forgotten-condition grid](samples_forget.png)', '',
          '[Guidance-2 retained-condition grid](samples_retain.png)', '']
(out/'comparison.md').write_text('\n'.join(lines))
print(out/'comparison.md')
