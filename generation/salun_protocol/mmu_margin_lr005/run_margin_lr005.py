import os
import copy, json, sys, time, traceback
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / 'mmu_variants_3seed'
sys.path.insert(0, str(STUDY))
import run as S

LR = 5e-3
TARGETS = ['V06', 'V07', 'V08', 'V09', 'V10', 'V11']
OUT = HERE


def main():
    torch.set_num_threads(4)
    variants = [v for v in S.V if v['id'] in TARGETS]
    assert len(variants) == len(TARGETS), [v['id'] for v in variants]
    S.dump(OUT / 'variants.json', variants)
    all_results = {}
    for mode in ['class', 'instance', 'subclass']:
        for seed in [42, 43, 44]:
            cell = f'{mode}/seed{seed}'
            S.dump(OUT / 'status.json', dict(status='running', cell=cell, lr=LR))
            b, old, out = S.setup(mode, seed)
            out = Path(os.environ.get('MMU_WORK','work')+'/mmu_margin_lr005') / mode / f'seed{seed}'
            out.mkdir(parents=True, exist_ok=True)
            best = json.loads((old / 'search.json').read_text())['best']
            core = {k: best['mmu']['config'][k] for k in ['lr', 'epochs', 'msteps']}
            core['max_grad_norm'] = 5.
            core['lr'] = LR
            rows = {}
            for name in ['Source', 'Retrain']:
                model = b._fresh_source() if name == 'Source' else copy.deepcopy(b.reference)
                model.eval()
                rows[name] = dict(id=name, name=name, mode=mode, seed=seed,
                                  metrics=b.evaluate(model, 'final'), config={})
            for v in variants:
                rid = v['id']
                result = out / f'{rid}.json'
                if result.exists():
                    rows[rid] = json.loads(result.read_text())
                    continue
                S.B.set_seed(seed)
                start = time.time()
                config = {**core, 'head_mode': v['head'], 'weighting': v['weight'],
                          'bad_margin': v['margin']}
                model, hist, aux = S.custom(b, v, core, None)
                torch.cuda.synchronize()
                seconds = time.time() - start
                model.eval()
                metrics = b.evaluate(model, 'final')
                assert S.B.finite_metrics(metrics)
                row = dict(id=rid, name=v['name'] + f' [lr {LR:g}]', mode=mode, seed=seed,
                           metrics=metrics, seconds=seconds, config=config)
                S.dump(result, row)
                rows[rid] = row
                print('RESULT', cell, rid, metrics, flush=True)
                all_results[cell] = rows
                S.dump(OUT / 'results.json', all_results)
            all_results[cell] = rows
            S.dump(OUT / 'results.json', all_results)
            del b
    S.dump(OUT / 'status.json', dict(status='complete', cells=9, variants=len(TARGETS), lr=LR))
    print('MARGIN LR005 COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        S.dump(OUT / 'status.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
