import sys, pickle
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from ddpm import Diffusion

import os
REPO = Path(__file__).resolve().parents[2]
BACKBONES = str(REPO / 'mmu/core/models/backbones.py')

CLS_CK = os.environ.get('MMU_JUDGE_CKPT', os.environ.get('MMU_WORK', 'work') +
                        '/cifar10/mu/resnet18_adam_class/seed42/cache/source.pt')
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def judge(dev='cuda'):
    import importlib.util
    spec = importlib.util.spec_from_file_location('mmu_backbones', BACKBONES)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.resnet18(num_classes=10, small_input=True)
    m.load_state_dict(torch.load(CLS_CK, map_location=dev, weights_only=False))
    return m.to(dev).eval()

@torch.no_grad()
def classify(imgs, clf, dev='cuda'):
    x = (imgs + 1) / 2
    x = (x - MEAN.to(dev)) / STD.to(dev)
    out = []
    for i in range(0, len(x), 500):
        out.append(clf(x[i:i+500]).argmax(1))
    return torch.cat(out)

@torch.no_grad()
def generate(model, diff, labels, bs=250, guidance=2.0, steps=100, dev='cuda'):
    outs = []
    for i in range(0, len(labels), bs):
        y = labels[i:i+bs].to(dev)
        outs.append(diff.sample(model, len(y), y, device=dev, guidance=guidance, steps=steps).cpu())
    return torch.cat(outs)

class FID:
    def __init__(self, dev='cuda'):
        from pytorch_fid.inception import InceptionV3
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
        self.dev = dev
    @torch.no_grad()
    def feats(self, imgs, bs=100):
        f = []
        for i in range(0, len(imgs), bs):
            x = ((imgs[i:i+bs] + 1) / 2).clamp(0, 1).to(self.dev)
            f.append(self.net(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(f)
    def score(self, a, b):
        from pytorch_fid.fid_score import calculate_frechet_distance
        fa, fb = self.feats(a), self.feats(b)
        return float(calculate_frechet_distance(fa.mean(0), np.cov(fa, rowvar=False),
                                                fb.mean(0), np.cov(fb, rowvar=False)))

def cifar10_test(dev='cuda'):
    with open(os.environ.get('MMU_DATA','data')+'/cifar-10-batches-py/test_batch', 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    x = torch.from_numpy(d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 127.5 - 1.0)
    return x, torch.tensor(d[b'labels'])

def evaluate(model, forget_class, *, n_ua=500, n_fid=2000, guidance=2.0, steps=100,
             dev='cuda', clf=None, fid=None, real=None):
    diff = Diffusion(device=dev)
    clf = clf or judge(dev)
    res = {}
    yf = torch.full((n_ua,), forget_class, dtype=torch.long)
    gf = generate(model, diff, yf, guidance=guidance, steps=steps, dev=dev)
    pred = classify(gf.to(dev), clf, dev).cpu()
    res['UA'] = 100.0 * float((pred != forget_class).float().mean())
    res['forget_class_rate'] = 100.0 * float((pred == forget_class).float().mean())
    keep = [c for c in range(10) if c != forget_class]
    yr = torch.tensor(np.random.default_rng(0).choice(keep, n_fid))
    gr = generate(model, diff, yr, guidance=guidance, steps=steps, dev=dev)
    predr = classify(gr.to(dev), clf, dev).cpu()
    res['retain_cond_acc'] = 100.0 * float((predr == yr).float().mean())
    if fid is not None and real is not None:
        res['FID_retain'] = fid.score(gr, real)
    return res, gf, gr
