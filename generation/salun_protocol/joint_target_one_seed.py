import os
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import classification as C
from gradient_pass_diagnostic import M, snapshot

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'joint_target_one_seed'
SEED = 2


def uniform_kl(logits):
    return -F.log_softmax(logits, dim=1).mean() - math.log(logits.shape[1])


def take(ds, indices):
    ds = copy.deepcopy(ds)
    ds.data = ds.data[indices]
    ds.targets = np.asarray(ds.targets)[indices]
    return ds


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    sys.argv = ['joint-target']
    args = C.arg_parser.parse_args()
    args.data = os.environ.get('MMU_DATA','data')
    args.seed = args.train_seed = SEED
    original = C.utils.cifar10_dataloaders
    def same_split(*v, **kw):
        kw.setdefault('seed', SEED)
        return original(*v, **kw)
    C.utils.cifar10_dataloaders = same_split
    source, tr, va, te, _ = C.utils.setup_model_dataset(args)
    source.cuda()
    base = ROOT / f'classification/seed{SEED}'
    source.load_state_dict(torch.load(base / 'source.pt', weights_only=False)['state_dict'])
    source.eval()
    split = np.load(base / 'forget10/split.npz')
    forget = take(tr.dataset, np.sort(split['forget']))
    retain = take(tr.dataset, split['retain'])
    indices = np.load(ROOT / 'diagnostic_one_seed/probe_indices.npz')
    probe_ds = {'forget': copy.deepcopy(forget),
                'retain_probe': take(retain, indices['retain']),
                'test_probe': take(te.dataset, indices['test'])}
    for ds in probe_ds.values():
        ds.transform = transforms.ToTensor()
    probes = {k: C.loader(ds) for k, ds in probe_ds.items()}
    config = dict(seed=SEED, ratio=.1, epochs=5, lr=.001, batch_size=256,
                  forget_weight=len(forget)/len(retain), temperature_retain=4,
                  forget_target='uniform', retain_teacher='full_width',
                  width_reduction='mean', controls=['fullwidth_joint', 'nested_joint'],
                  checkpoint_rule='final epoch, fixed before training',
                  note='Exploratory surrogate, not exact unlearning or a SCRUB reproduction. '
                       'Multiple changes vs archived MMU; only width differs between new controls. '
                       'Forget data are recycled once per retain batch, increasing exposures.')
    C.dump(OUT / 'config.json', config)
    paths = [ROOT/'classification_summary.md',
             ROOT/'diagnostic_one_seed/report.md', Path(__file__)]
    C.dump(OUT/'input_sha256.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    z = torch.tensor([[5., 0., 0.]], requires_grad=True)
    loss = uniform_kl(z); loss.backward()
    assert z.grad[0, 0] > 0 and z.grad[0, 1] < 0
    u = torch.zeros(2, 10, requires_grad=True)
    uniform_kl(u).backward()
    assert u.grad.abs().max() < 1e-7 and abs(float(uniform_kl(u))) < 1e-6
    results = {}
    for name, full in [('fullwidth_joint', True), ('nested_joint', False)]:
        C.utils.setup_seed(SEED)
        student = copy.deepcopy(source)
        kw = {'granularities': [M.linear_head(student).in_features]} if full else {}
        method = M.MSCRUB(student, diagnose=False, **kw)
        method.heads = M.matryoshka_heads(method.head, method.granularities, 'cuda')
        teacher = M.clone_frozen(source)
        params = list(student.parameters()) + [p for h in method.heads.values() for p in h.parameters()]
        opt = torch.optim.SGD(params, lr=config['lr'], momentum=.9, weight_decay=5e-4)
        C.utils.setup_seed(SEED)
        history = [snapshot(method, teacher, probes, 'source')]
        train_seconds = 0.
        for epoch in range(config['epochs']):
            start = time.perf_counter()
            student.train()
            fd = C.loader(forget, True)
            fi = iter(fd)
            losses = np.zeros(3)
            steps = 0
            for xr, yr in C.loader(retain, True):
                try:
                    xf, _ = next(fi)
                except StopIteration:
                    fi = iter(fd)
                    xf, _ = next(fi)
                xr, yr, xf = xr.cuda(), yr.cuda(), xf.cuda()
                with torch.no_grad():
                    target = teacher(xr)
                logits = method._student_logits(torch.cat([xr, xf]))
                retain_loss = sum(F.cross_entropy(s[:len(xr)], yr) +
                                  M.kd_kl(s[:len(xr)], target, 4.) for s in logits) / len(logits)
                forget_loss = sum(uniform_kl(s[len(xr):]) for s in logits) / len(logits)
                loss = retain_loss + config['forget_weight'] * forget_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite objective')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses += [float(loss.detach()), float(retain_loss.detach()), float(forget_loss.detach())]
                steps += 1
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - start
            record = snapshot(method, teacher, probes, f'epoch{epoch}_joint')
            record['losses'] = dict(zip(['total', 'retain', 'forget'], (losses/steps).tolist()))
            record['steps'] = steps
            history.append(record)
            C.dump(OUT/f'{name}_history.json', history)
            print(name, epoch, record['sets']['forget']['512'], flush=True)
        assert all(p.grad is None for p in teacher.parameters())
        assert all(torch.equal(v, source.state_dict()[k]) for k, v in teacher.state_dict().items())
        assert all(torch.isfinite(p).all() for p in student.parameters())
        torch.save({'state_dict': student.state_dict(), 'auxiliary_heads':
                    {m: h.state_dict() for m, h in method.heads.items()}, 'config': config}, OUT/f'{name}.pt')
        result = C.evaluate(student, retain, forget, te.dataset)
        result.update(RTE_min=train_seconds/60, seed=SEED, method=name,
                      widths=list(method.granularities))
        C.dump(OUT/f'{name}.json', result)
        results[name] = result
        print('FINAL', name, result, flush=True)
    C.dump(OUT/'results.json', results)
    print('ONE-SEED JOINT TARGET COMPLETE', flush=True)


if __name__ == '__main__':
    main()
