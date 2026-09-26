import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

import run
from run import nested, predict


def main():
    torch.manual_seed(0)
    u = UNet2DConditionModel.from_pretrained(run.SOURCE, subfolder='unet').cuda().eval()
    e = run.Engine.__new__(run.Engine)
    e.unet = u
    from transformers import CLIPTokenizer, CLIPTextModel
    e.tokenizer = CLIPTokenizer.from_pretrained(run.SOURCE/'tokenizer')
    e.text = CLIPTextModel.from_pretrained(run.SOURCE/'text_encoder').cuda().eval()
    e.embeddings = {}
    emb = e.embed([run.RETAIN])
    x = torch.randn(1, 4, 64, 64, device='cuda')
    t = torch.tensor([500], device='cuda')

    with torch.no_grad():
        full = predict(u, x, t, emb)
        assert torch.equal(full, nested(u, x, t, emb, [320])[0]), 'width 320 must equal plain forward'
        print('OK  width 320 == plain forward')

        for m in [80, 160]:
            base = nested(u, x, t, emb, [m])[0]

            def corrupt(module, args, m=m):
                h = args[0].clone()
                h[:, m:] += 25.0 * torch.randn_like(h[:, m:])
                return (h,)

            handle = u.conv_norm_out.register_forward_pre_hook(corrupt)
            try:
                perturbed = nested(u, x, t, emb, [m])[0]
            finally:
                handle.remove()
            drift = (perturbed - base).abs().max().item()
            assert drift < 1e-4, f'width {m} leaks discarded channels, max drift {drift}'
            print(f'OK  width {m} independent of channels {m}:320 (max drift {drift:.2e})')

        for m in [80, 160]:
            d = (nested(u, x, t, emb, [m])[0] - full).abs().mean().item()
            assert d > 1e-3, f'width {m} is indistinguishable from full width'
            print(f'OK  width {m} differs from full width (mean |delta| {d:.4f})')

    print('\nall nested-prefix checks passed')


if __name__ == '__main__':
    main()
