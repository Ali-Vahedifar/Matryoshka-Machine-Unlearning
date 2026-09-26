import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from ddpm import CondUNet, Diffusion
from train_source import cifar10
import unlearn as U, evaluate as E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forget_class', type=int, default=0)
    ap.add_argument('--ckpt', default='diffusion_mmu/ckpt/source.pt')
    ap.add_argument('--retrain_ckpt', default='diffusion_mmu/ckpt/retrain_c0.pt')
    ap.add_argument('--out', default='diffusion_mmu/results')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--n_ua', type=int, default=500)
    ap.add_argument('--n_fid', type=int, default=5000)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--guidance', type=float, default=1.0)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--final_repair', type=int, default=0)
    ap.add_argument('--warmup_epochs', type=int, default=0)
    ap.add_argument('--retain_frac', type=float, default=0.3)
    ap.add_argument('--methods', default='mmu_fullwidth,mmu')
    ap.add_argument('--tag', default='redirect_v1')
    ap.add_argument('--objective', choices=['redirect', 'repulsion'], default='redirect')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--null_weight', type=float, default=0.1)
    ap.add_argument('--eval_checkpoint', help='Evaluate one saved model without training')
    a = ap.parse_args()
    dev = 'cuda'; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allowed = {'source', 'retrain', 'finetune', 'neggrad', 'salun',
               'mmu', 'mmu_noskip', 'mmu_retainonly', 'mmu_fullwidth'}
    names = a.methods.split(',')
    if any(n not in allowed for n in names) or len(names) != len(set(names)):
        ap.error('Methods must be unique supported names')
    if a.eval_checkpoint and len(names) != 1:
        ap.error('--eval_checkpoint requires exactly one --methods name')
    if not 0 < a.retain_frac <= 1 or not 0 <= a.forget_class < 10:
        ap.error('Invalid retain fraction or forget class')
    if a.batch <= 0 or a.epochs < 0 or a.n_ua < 1 or a.n_fid < 2 or a.steps < 2:
        ap.error('Invalid batch, epoch, sample count, or sampling steps')
    run = out / f'{a.tag}_c{a.forget_class}_seed{a.seed}'
    run.mkdir(exist_ok=False)
    (run / 'config.json').write_text(json.dumps(vars(a), indent=2))
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    x, y = cifar10(); x, y = x.to(dev), y.to(dev)
    fmask = y == a.forget_class
    xf, yf = x[fmask], y[fmask]
    xr_all, yr_all = x[~fmask], y[~fmask]
    k = int(len(xr_all) * a.retain_frac)
    sel = torch.randperm(len(xr_all), device=dev)[:k]
    xr, yr = xr_all[sel], yr_all[sel]
    print(f'forget class {a.forget_class} ({E.CLASSES[a.forget_class]}): '
          f'|D_f|={len(xf)}  |D_r used|={len(xr)} of {len(xr_all)}', flush=True)

    diff = Diffusion(device=dev)
    clf = E.judge(dev); fid = E.FID(dev)
    xt, yt = E.cifar10_test()
    real_retain = xt[yt != a.forget_class][:a.n_fid]
    print(f'FID reference: {len(real_retain)} real retain-class test images', flush=True)

    results, grids, diagnostics = {}, {}, {}
    for name in names:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        t0 = time.time()
        print(f'\n=== {name} ===', flush=True)
        if a.eval_checkpoint:
            model = U.load_source(a.eval_checkpoint, dev)
        elif name == 'retrain':
            if not Path(a.retrain_ckpt).exists():
                print('  retrain checkpoint missing, skipping'); continue
            model = U.load_source(a.retrain_ckpt, dev)
        else:
            model = U.load_source(a.ckpt, dev)
        if a.eval_checkpoint:
            pass
        elif name.startswith('mmu'):
            variant = {'mmu': 'full', 'mmu_noskip': 'full',
                       'mmu_retainonly': 'retain_only',
                       'mmu_fullwidth': 'full_width'}[name]
            teacher = U.load_source(a.ckpt, dev).eval()
            for p in teacher.parameters(): p.requires_grad_(False)
            model, mhist = U.mmu_gen(model, teacher, xr, yr, xf, yf, diff, epochs=a.epochs,
                                     delta=a.delta, trunc=(name != 'mmu_noskip'),
                                     variant=variant, final_repair=a.final_repair, dev=dev,
                                     objective=a.objective, beta=a.beta, null_weight=a.null_weight,
                                     warmup_epochs=a.warmup_epochs,
                                     alpha=a.alpha, gamma=a.gamma, lr=a.lr, bs=a.batch,
                                     log=lambda s: print(s, flush=True))
            diagnostics[name] = mhist
        elif name == 'finetune':
            model = U.finetune(model, xr, yr, diff, epochs=a.epochs, lr=a.lr, bs=a.batch, dev=dev,
                               log=lambda s: print(s, flush=True))
        elif name == 'neggrad':
            model = U.neggrad(model, xr, yr, xf, yf, diff, epochs=a.epochs, lr=a.lr, bs=a.batch, dev=dev,
                              log=lambda s: print(s, flush=True))
        elif name == 'salun':
            model = U.salun(model, xr, yr, xf, yf, diff, epochs=a.epochs, lr=a.lr, bs=a.batch, dev=dev,
                            log=lambda s: print(s, flush=True))
        unlearn_time = time.time() - t0
        if not a.eval_checkpoint:
            torch.save({'model': model.state_dict(), 'base': model.stem.out_channels,
                        'config': vars(a), 'method': name}, run / f'{name}.pt')
        (run / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        torch.manual_seed(a.seed + 10000); np.random.seed(a.seed + 10000)
        res, gf, gr = E.evaluate(model, a.forget_class, n_ua=a.n_ua, n_fid=a.n_fid,
                                 guidance=a.guidance, steps=a.steps, dev=dev,
                                 clf=clf, fid=fid, real=real_retain)
        res['unlearn_time_s'] = round(unlearn_time, 1)
        results[name] = res
        grids[name] = {'forget': gf[:16].clone(), 'retain': gr[:16].clone()}
        print(f'  -> UA={res["UA"]:.1f}%  forget-rate={res["forget_class_rate"]:.1f}%  '
              f'FID={res.get("FID_retain",float("nan")):.2f}  '
              f'retain-cond-acc={res["retain_cond_acc"]:.1f}%  t={unlearn_time:.0f}s', flush=True)
        json.dump(results, open(run / 'results.json', 'w'), indent=1)
        if diagnostics:
            json.dump(diagnostics, open(run / 'diagnostics.json', 'w'),
                      indent=1, default=str)
        torch.save(grids, run / 'grids.pt')
    print('\nDONE', json.dumps(results, indent=1), flush=True)

if __name__ == '__main__':
    main()
