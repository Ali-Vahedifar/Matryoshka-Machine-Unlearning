import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from ddpm import CondUNet, Diffusion
from train_source import cifar10, EMA

def load_source(ck, dev='cuda', use_ema=True):
    d = torch.load(ck, map_location=dev, weights_only=False)
    m = CondUNet(base=d['base']).to(dev)
    m.load_state_dict(d['ema'] if use_ema and 'ema' in d else d['model'])
    return m

def batches(x, y, bs, shuffle=True):
    idx = torch.randperm(len(x), device=x.device) if shuffle else torch.arange(len(x), device=x.device)
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]; yield x[j], y[j]

def eps_at(model, xt, t, y, frac, trunc):
    return model(xt, t, y, prefix_frac=frac, truncate_skips=trunc)

def discrepancy(a, b):
    return (a - b).pow(2).flatten(1).mean(1)


@torch.no_grad()
def retain_probe(model, xr, yr, diff, fracs, trunc, dev, n=1024, seed=0):
    was_training = model.training
    model.eval()
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.randperm(len(xr), device=dev, generator=g)[:n]
    xb, yb = xr[idx], yr[idx]
    t = torch.randint(0, diff.T, (len(xb),), device=dev, generator=g)
    noise = torch.randn(xb.shape, device=dev, generator=g)
    xt = diff.q_sample(xb, t, noise)
    out = {f: float(F.mse_loss(eps_at(model, xt, t, yb, f, trunc), noise)) for f in fracs}
    if was_training:
        model.train()
    return out


def mmu_repulsion(model, teacher, xr, yr, xf, yf, diff, *, fracs=(0.25, 0.5, 1.0),
            alpha=1.0, gamma=1.0, delta=0.5, lr=1e-4, epochs=4, bs=128,
            trunc=True, variant='full', final_repair=0, dev='cuda', log=print):
    if variant == 'full_width':
        fracs = (1.0,)
    lam_r = {f: 1.0 for f in fracs}
    lam_f = {f: 1.0 for f in fracs}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    hist = []
    probe0 = retain_probe(model, xr, yr, diff, fracs, trunc, dev)
    log(f'  [{variant}] retain probe @start: ' +
        '  '.join(f'm={f}:{v:.4f}' for f, v in probe0.items()))
    for ep in range(epochs):
        model.train()
        r_tot = r_n = 0
        for xb, yb in batches(xr, yr, bs):
            t = torch.randint(0, diff.T, (len(xb),), device=dev)
            noise = torch.randn_like(xb); xt = diff.q_sample(xb, t, noise)
            with torch.no_grad():
                tch = {f: eps_at(teacher, xt, t, yb, f, trunc) for f in fracs}
            loss = 0.0
            for f in fracs:
                e = eps_at(model, xt, t, yb, f, trunc)
                loss = loss + lam_r[f] * (gamma * F.mse_loss(e, noise)
                                          + alpha * F.mse_loss(e, tch[f]))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            r_tot += float(loss); r_n += 1
        probe_r = retain_probe(model, xr, yr, diff, fracs, trunc, dev)

        f_tot = f_n = 0
        d_sum = {f: 0.0 for f in fracs}
        hinge_on = {f: 0 for f in fracs}
        seen = 0
        if variant != 'retain_only':
            for xb, yb in batches(xf, yf, bs):
                t = torch.randint(0, diff.T, (len(xb),), device=dev)
                noise = torch.randn_like(xb); xt = diff.q_sample(xb, t, noise)
                with torch.no_grad():
                    tch = {f: eps_at(teacher, xt, t, yb, f, trunc) for f in fracs}
                loss = 0.0
                for f in fracs:
                    e = eps_at(model, xt, t, yb, f, trunc)
                    d_m = discrepancy(e, tch[f])
                    loss = loss + lam_f[f] * torch.clamp(delta - d_m, min=0).mean()
                    d_sum[f] += float(d_m.sum())
                    hinge_on[f] += int((d_m < delta).sum())
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                f_tot += float(loss); f_n += 1; seen += len(xb)
        probe_f = retain_probe(model, xr, yr, diff, fracs, trunc, dev)

        rec = {'epoch': ep,
               'retain_loss': r_tot / max(r_n, 1),
               'forget_loss': f_tot / max(f_n, 1),
               'retain_batches': r_n, 'forget_batches': f_n,
               'd_mean': {f: (d_sum[f] / seen if seen else None) for f in fracs},
               'hinge_active': {f: (hinge_on[f] / seen if seen else None) for f in fracs},
               'probe_after_retain': probe_r, 'probe_after_forget': probe_f}
        hist.append(rec)
        dm = '  '.join(f"m={f}:d={rec['d_mean'][f]:.4f}/on={rec['hinge_active'][f]:.2f}"
                       if rec['d_mean'][f] is not None else f'm={f}:--' for f in fracs)
        pr = '  '.join(f'm={f}:{probe_r[f]:.4f}->{probe_f[f]:.4f}' for f in fracs)
        log(f'  [{variant}] ep{ep}: L_r={rec["retain_loss"]:.4f} '
            f'L_f={rec["forget_loss"]:.4f} | {dm} | probe(afterR->afterF) {pr}')

    for rp in range(final_repair):
        model.train(); r_tot = r_n = 0
        for xb, yb in batches(xr, yr, bs):
            t = torch.randint(0, diff.T, (len(xb),), device=dev)
            noise = torch.randn_like(xb); xt = diff.q_sample(xb, t, noise)
            with torch.no_grad():
                tch = {f: eps_at(teacher, xt, t, yb, f, trunc) for f in fracs}
            loss = 0.0
            for f in fracs:
                e = eps_at(model, xt, t, yb, f, trunc)
                loss = loss + lam_r[f] * (gamma * F.mse_loss(e, noise)
                                          + alpha * F.mse_loss(e, tch[f]))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            r_tot += float(loss); r_n += 1
        pb = retain_probe(model, xr, yr, diff, fracs, trunc, dev)
        hist.append({'epoch': f'repair{rp}', 'retain_loss': r_tot / max(r_n, 1),
                     'probe_after_retain': pb})
        log(f'  [{variant}] repair{rp}: L_r={r_tot/max(r_n,1):.4f} | probe ' +
            '  '.join(f'm={f}:{v:.4f}' for f, v in pb.items()))
    return model, hist


def finetune(model, xr, yr, diff, *, lr=1e-4, epochs=4, bs=128, dev='cuda', log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        model.train(); tot = nb = 0
        for xb, yb in batches(xr, yr, bs):
            t = torch.randint(0, diff.T, (len(xb),), device=dev)
            n = torch.randn_like(xb)
            loss = F.mse_loss(model(diff.q_sample(xb, t, n), t, yb), n)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        log(f'  finetune ep{ep}: {tot/max(nb,1):.4f}')
    return model

def neggrad(model, xr, yr, xf, yf, diff, *, lr=5e-5, epochs=4, bs=128, dev='cuda', log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        model.train(); tot = nb = 0
        fb = list(batches(xf, yf, bs))
        for i, (xb, yb) in enumerate(batches(xr, yr, bs)):
            t = torch.randint(0, diff.T, (len(xb),), device=dev); n = torch.randn_like(xb)
            loss = F.mse_loss(model(diff.q_sample(xb, t, n), t, yb), n)
            if fb:
                xf_, yf_ = fb[i % len(fb)]
                tf = torch.randint(0, diff.T, (len(xf_),), device=dev); nf = torch.randn_like(xf_)
                loss = loss - F.mse_loss(model(diff.q_sample(xf_, tf, nf), tf, yf_), nf)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        log(f'  neggrad ep{ep}: {tot/max(nb,1):.4f}')
    return model

def salun(model, xr, yr, xf, yf, diff, *, lr=1e-4, epochs=4, bs=128, sparsity=0.5,
          num_classes=10, dev='cuda', log=print):
    model.train(); model.zero_grad(set_to_none=True)
    for xb, yb in batches(xf, yf, bs):
        t = torch.randint(0, diff.T, (len(xb),), device=dev); n = torch.randn_like(xb)
        F.mse_loss(model(diff.q_sample(xb, t, n), t, yb), n).backward()
    grads = {k: p.grad.abs().flatten() for k, p in model.named_parameters() if p.grad is not None}
    allg = torch.cat(list(grads.values()))
    thr = torch.quantile(allg[torch.randperm(len(allg), device=dev)[:2_000_000]], 1 - sparsity)
    mask = {k: (p.grad.abs() >= thr).float() for k, p in model.named_parameters() if p.grad is not None}
    kept = sum(m.sum().item() for m in mask.values()) / sum(m.numel() for m in mask.values())
    log(f'  salun mask keeps {100*kept:.1f}% of weights')
    model.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        tot = nb = 0
        fb = list(batches(xf, yf, bs))
        for i, (xb, yb) in enumerate(batches(xr, yr, bs)):
            t = torch.randint(0, diff.T, (len(xb),), device=dev); n = torch.randn_like(xb)
            loss = F.mse_loss(model(diff.q_sample(xb, t, n), t, yb), n)
            if fb:
                xf_, yf_ = fb[i % len(fb)]
                tf = torch.randint(0, diff.T, (len(xf_),), device=dev)
                nf = torch.randn_like(xf_)
                xtf = diff.q_sample(xf_, tf, nf)
                loss = loss + salun_prediction_loss(model, xtf, tf, yf_, num_classes)
            opt.zero_grad(set_to_none=True); loss.backward()
            for k, p in model.named_parameters():
                if p.grad is not None and k in mask: p.grad.mul_(mask[k])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        log(f'  salun ep{ep}: {tot/max(nb,1):.4f}')
    return model


def salun_prediction_loss(model, xt, t, labels, num_classes):
    with torch.no_grad():
        target = model(xt, t, (labels + 1) % num_classes)
    return F.mse_loss(model(xt, t, labels), target)


def mmu_gen(model, teacher, xr, yr, xf, yf, diff, *, fracs=(0.25, 0.5, 1.0),
            alpha=1.0, gamma=1.0, delta=0.05, lr=1e-4, epochs=3, bs=128,
            trunc=True, variant='full_width', final_repair=0, dev='cuda', log=print,
            objective='redirect', beta=1.0, null_weight=0.1, warmup_epochs=0):
    if variant not in ('full', 'full_width', 'retain_only'):
        raise ValueError('Unknown MMU variant')
    if objective not in ('redirect', 'repulsion'):
        raise ValueError('Unknown MMU objective')
    if bs <= 0 or epochs < 0 or final_repair < 0 or warmup_epochs < 0 or lr <= 0:
        raise ValueError('Invalid optimization settings')
    if min(alpha, gamma, beta, null_weight) < 0 or not len(xr) or not len(yf):
        raise ValueError('Nonnegative weights and nonempty retain/forget sets required')
    if torch.isin(yr, yf.unique()).any():
        raise ValueError('Retained and forgotten labels must be disjoint')
    teacher.eval()
    teacher.requires_grad_(False)
    if objective == 'repulsion':
        if warmup_epochs:
            raise ValueError('Prefix warm-up is supported only for redirection')
        return mmu_repulsion(model, teacher, xr, yr, xf, yf, diff, fracs=fracs,
            alpha=alpha, gamma=gamma, delta=delta, lr=lr, epochs=epochs, bs=bs,
            trunc=trunc, variant=variant, final_repair=final_repair, dev=dev, log=log)
    if variant == 'full_width':
        fracs = (1.0,)
    if not fracs or any(f <= 0 or f > 1 for f in fracs):
        raise ValueError('Widths must be in (0, 1]')
    forget_labels = yf.unique()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    hist = []
    for ep in range(warmup_epochs + epochs + final_repair):
        model.train()
        warmup = ep < warmup_epochs
        repair = ep >= warmup_epochs + epochs
        totals = dict(retain_loss=0.0, redirect_loss=0.0, null_loss=0.0)
        width_error = {f: 0.0 for f in fracs}
        seen = updates = 0
        for xb, yb in batches(xr, yr, bs):
            n = len(xb)
            t = torch.randint(0, diff.T, (n,), device=dev)
            noise = torch.randn_like(xb)
            xt = diff.q_sample(xb, t, noise)
            cf = forget_labels[torch.randint(len(forget_labels), (n,), device=dev)]
            null = torch.full_like(yb, model.num_classes)
            with torch.no_grad():
                target = eps_at(teacher, xt, t, yb, 1.0, trunc)
                null_target = teacher(xt, t, null) if null_weight else None
            rloss = xt.new_zeros(())
            floss = xt.new_zeros(())
            for f in fracs:
                pred = eps_at(model, xt, t, yb, f, trunc)
                rloss = rloss + (gamma * F.mse_loss(pred, noise)
                                  + alpha * F.mse_loss(pred, target)) / len(fracs)
                if not warmup and not repair and variant != 'retain_only':
                    err = discrepancy(eps_at(model, xt, t, cf, f, trunc), target)
                    floss = floss + err.mean() / len(fracs)
                    width_error[f] += float(err.detach().sum())
            nloss = F.mse_loss(model(xt, t, null), null_target) if null_weight else xt.new_zeros(())
            loss = rloss + beta * floss + null_weight * nloss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite MMU loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for key, val in zip(totals, (rloss, floss, nloss)):
                totals[key] += float(val.detach()) * n
            seen += n
            updates += 1
        rec = {key: val / seen for key, val in totals.items()}
        rec.update(epoch=ep, phase='warmup' if warmup else ('repair' if repair else 'joint'), updates=updates,
                   redirect_mse={f: width_error[f] / seen for f in fracs},
                   retain_probe=retain_probe(model, xr, yr, diff, fracs, trunc, dev))
        hist.append(rec)
        log(f'  [{variant}/redirect] ep{ep}: {rec}')
    return model, hist
