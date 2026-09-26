import os
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'joint_target_one_seed'
BASE = ROOT/'classification/seed2/forget10'
METRICS = ['UA', 'RA', 'TA', 'MIA']


def main():
    rows = [(label, json.loads(path.read_text())) for label, path in [
        ('Retrain', BASE/'retrain.json'),
        ('SalUn (released-code adapter)', BASE/'SalUn.json'),
        ('MMU (previous adapter)', BASE/'MMU.json'),
        ('MMU (transferred old recipe)', ROOT/'old_recipe_one_seed/results.json'),
        ('Joint uniform target, full width', OUT/'fullwidth_joint.json'),
        ('Joint uniform target, nested', OUT/'nested_joint.json')]]
    gold = rows[0][1]
    for _, row in rows:
        row['gaps'] = {m: abs(row[m]-gold[m]) for m in METRICS}
        row['avg_gap'] = sum(row['gaps'].values())/len(METRICS)
    C = {label: row for label, row in rows}
    (OUT/'comparison.json').write_text(json.dumps(C, indent=2))
    lines = ['# One-seed MMU improvement attempt', '',
        'Completed: seed 2, CIFAR-10, 10% random forgetting. Three new fixed training runs: '
        'two matched joint-target controls and one transfer of the archived MMU recipe. '
        'No sweep, extra seeds or source retraining. Existing baseline rows are reused from this same seed and split.', '',
        'All four accuracy-related metrics should approach Retrain; D_f is forget accuracy (100 minus UA), D_r is retain accuracy, and Test is test accuracy. '
        'Parentheses show absolute gaps to Retrain in percentage points. Avg. gap and runtime should decrease.', '',
        '| Method | D_f accuracy → Retrain | D_r accuracy → Retrain | Test accuracy → Retrain | MIA → Retrain | Avg. gap ↓ | Train min ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label, r in rows:
        values = [f'{(100-r[m] if m=="UA" else r[m]):.2f} ({r["gaps"][m]:.2f})' for m in METRICS]
        lines.append('| '+label+' | '+' | '.join(values)+f' | {r["avg_gap"]:.2f} | {r["RTE_min"]:.2f} |')
    old, nested, full, salun = [C[k] for k in ['MMU (previous adapter)', 'Joint uniform target, nested',
                                             'Joint uniform target, full width', 'SalUn (released-code adapter)']]
    lines += ['', '## Interpretation', '',
        f'The joint nested variant changes the average gap from {old["avg_gap"]:.2f} to '
        f'{nested["avg_gap"]:.2f}. Its D_f accuracy is {100-nested["UA"]:.2f}% against Retrain {100-gold["UA"]:.2f}%, '
        f'and its MIA is {nested["MIA"]:.2f}% against {gold["MIA"]:.2f}%. '
        'These values must be considered together: preserving accuracy does not establish removal of data influence.', '',
        f'The full-width joint control has average gap {full["avg_gap"]:.2f}; the nested version has '
        f'{nested["avg_gap"]:.2f}. This is a single-seed comparison, not statistical evidence of a general nesting advantage. '
        f'The SalUn adapter has average gap {salun["avg_gap"]:.2f}.', '',
        'In this pilot, the nested revision improves the aggregate gap while preserving utility, '
        'but still underforgets and remains behind SalUn. This is partial progress, not a solved method. '
        'The archived recipe transfer severely damages the current model (retained accuracy about 21%). '
        'Restoring old settings therefore does not recover old performance on this source. '
        'The schedule mismatch is real, but it is not the whole explanation.', '',
        'The transferred recipe is reported even if it damages utility. It uses the old seed-42 instance recipe '
        'unchanged: lr 0.005, five epochs, three forget passes, tied heads, summed width losses, '
        'max-step gradient clipping at 5, batch size 64 and a fixed 10% retained subset. '
        'Only the source/split/evaluation come from the current seed-2 protocol. It is a recipe-transfer check, '
        'not an ablation isolating the retain-pass ratio.', '',
        '## What changed in the experimental revision', '',
        'Each update jointly minimizes retained cross-entropy, retained full-width-teacher distillation '
        '(temperature 4), and KL from a uniform class target to the student on forgotten images '
        '(temperature 1). The forget coefficient is |Df|/|Dr| = 1/9, fixed before training. '
        'Losses are averaged across widths. The full-width teacher supervises every retained prefix; '
        'no assumption is made that frozen source prefixes are competent classifiers.', '',
        'The uniform target gives a finite optimum, unlike unrestricted repulsion. The KL loss itself '
        'is not bounded above. Uniform predictions are also not the true retraining target for random deletion; '
        'this is an exploratory confidence-reduction surrogate, not a certified unlearning objective. '
        'It changes the original MMU formulation and must be named separately in a paper.', '',
        'Both groups share each forward batch and optimizer step. There are 795 joint steps, with forgotten '
        'batches recycled. This increases forget exposure relative to the archived 36 forget steps, so '
        'the comparison is not compute matched. Only width differs between the two new joint controls. '
        'Final epoch five is used without test-based checkpoint selection. Runtime excludes diagnostic probes '
        'and final evaluation.', '',
        '## Verification and limits', '',
        'Checked finite losses and parameters, immutability of the frozen teacher including buffers, '
        'absence of teacher gradients, and the forget objective gradient at a confident prediction and its '
        'uniform optimum. Evaluated all 4,500 forgotten, 40,500 retained and 10,000 test images with the '
        'existing confidence-SVC evaluator. Epoch diagnostics reuse the prior fixed 2,048 retained/test probes. '
        'Original table and diagnostic contents are preserved and hash-verified in notation_archive_ua; current displays use D_f, D_r and Test accuracy. Auxiliary heads and final '
        'model weights are saved. No paper superiority claim follows from this pilot.', '',
        '[Why the earlier ranking changed](protocol_audit.md) · [Raw comparison](comparison.json) · '
        '[Fixed configuration](config.json)', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    for src, dst in [(os.environ.get('MMU_WORK', 'work') + '/mmu_joint_target_one_seed.log', OUT/'run.log'),
                     (os.environ.get('MMU_WORK', 'work') + '/mmu_old_recipe_one_seed.log', ROOT/'old_recipe_one_seed/run.log')]:
        shutil.copyfile(src, dst)
    import hashlib
    hashes = json.loads((OUT/'input_sha256.json').read_text())
    for p, h in hashes.items():
        current = Path(p)
        archived = ROOT/'notation_archive_ua'/current.relative_to(ROOT)
        assert any(q.exists() and hashlib.sha256(q.read_bytes()).hexdigest() == h for q in [current, archived])
    print('\n'.join(lines[:16]))


if __name__ == '__main__':
    main()
