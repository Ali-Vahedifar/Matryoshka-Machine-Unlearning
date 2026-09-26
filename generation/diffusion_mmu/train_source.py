import os
import argparse, copy, json, pickle, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from ddpm import CondUNet, Diffusion

def cifar10(root=os.environ.get('MMU_DATA','data')+'/cifar-10-batches-py'):
    xs, ys = [], []
    for i in range(1, 6):
        with open(f'{root}/data_batch_{i}', 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        xs.append(d[b'data']); ys += d[b'labels']
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(x), torch.tensor(ys)

class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay; self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()): s.copy_(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=220)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=64)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'ckpt'))
    ap.add_argument('--exclude_class', type=int, default=-1,
                    help='drop this class from training (retrain-from-scratch oracle)')
    ap.add_argument('--tag', default='source')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    torch.manual_seed(42); np.random.seed(42)

    x, y = cifar10()
    if a.exclude_class >= 0:
        keep = y != a.exclude_class
        x, y = x[keep], y[keep]
        print(f'excluding class {a.exclude_class}: {len(x)} images remain', flush=True)
    x, y = x.to(dev), y.to(dev)
    n = len(x)

    model = CondUNet(base=a.base).to(dev)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    diff = Diffusion(device=dev)
    steps_per_epoch = n // a.batch
    print(f'{n} imgs · {sum(p.numel() for p in model.parameters())/1e6:.1f}M params · '
          f'{steps_per_epoch} steps/epoch', flush=True)

    t0 = time.time(); hist = []
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i*a.batch:(i+1)*a.batch]
            x0 = x[idx]; yb = y[idx].clone()
            if torch.rand(1).item() < 1.1:
                drop = torch.rand(len(yb), device=dev) < model.dropout_cond
                yb[drop] = model.num_classes
            if torch.rand(1).item() < 0.5:
                x0 = torch.flip(x0, dims=[3])
            t = torch.randint(0, diff.T, (len(x0),), device=dev)
            noise = torch.randn_like(x0)
            loss = F.mse_loss(model(diff.q_sample(x0, t, noise), t, yb), noise)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model); tot += loss.item()
        hist.append(tot / steps_per_epoch)
        if ep % 5 == 0 or ep == a.epochs - 1:
            el = time.time() - t0
            print(f'epoch {ep:4d}/{a.epochs}  loss {hist[-1]:.4f}  '
                  f'{el/60:.1f} min  eta {el/(ep+1)*(a.epochs-ep-1)/60:.0f} min', flush=True)
            torch.save({'model': model.state_dict(), 'ema': ema.shadow.state_dict(),
                        'base': a.base, 'epoch': ep, 'loss_hist': hist,
                        'exclude_class': a.exclude_class}, out / f'{a.tag}.pt')
    json.dump(hist, open(out / f'{a.tag}_loss.json', 'w'))
    print('done', flush=True)

if __name__ == '__main__':
    main()
