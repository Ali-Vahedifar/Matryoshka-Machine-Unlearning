import copy
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from ddpm import CondUNet, Diffusion
from unlearn import mmu_gen, salun_prediction_loss, load_source, batches


class LabelModel(nn.Module):
    num_classes = 2
    def __init__(self):
        super().__init__()
        self.values = nn.Parameter(torch.tensor([5., 1., 0.]))
    def forward(self, x, t, y, prefix_frac=1., truncate_skips=True):
        return self.values[y, None, None, None].expand_as(x)


class RevisionTests(unittest.TestCase):
    def train_model(self, variant='full_width'):
        model = LabelModel()
        teacher = copy.deepcopy(model)
        xr = torch.zeros(3, 1, 2, 2)
        yr = torch.ones(3, dtype=torch.long)
        torch.manual_seed(123)
        _, hist = mmu_gen(model, teacher, xr, yr, xr[:1], yr[:1] * 0,
                          Diffusion(T=10, device='cpu'), variant=variant,
                          epochs=2, bs=2, gamma=0, lr=.1, dev='cpu', log=lambda s: None)
        return model, teacher, hist

    def test_redirection_changes_forget_toward_retained_target_only(self):
        model, teacher, hist = self.train_model()
        self.assertLess(abs(model.values[0].item() - 1), 4)
        torch.testing.assert_close(model.values[1:], teacher.values[1:])
        torch.testing.assert_close(teacher.values, torch.tensor([5., 1., 0.]))
        self.assertIsNone(teacher.values.grad)
        self.assertEqual(hist[0]['updates'], 2)
        self.assertLess(hist[-1]['redirect_loss'], hist[0]['redirect_loss'])

    def test_identical_prefixes_do_not_triple_loss_scale(self):
        full, _, h1 = self.train_model('full_width')
        nested, _, h2 = self.train_model('full')
        torch.testing.assert_close(full.values, nested.values)
        self.assertAlmostEqual(h1[0]['redirect_loss'], h2[0]['redirect_loss'], places=5)

    def test_salun_gradient_flows_to_forget_not_detached_target(self):
        model = LabelModel()
        loss = salun_prediction_loss(model, torch.zeros(2, 1, 2, 2),
                                     torch.zeros(2, dtype=torch.long),
                                     torch.zeros(2, dtype=torch.long), 2)
        loss.backward()
        self.assertGreater(model.values.grad[0].item(), 0)
        self.assertEqual(model.values.grad[1].item(), 0)

    def test_warmup_excludes_redirection_and_precedes_joint_updates(self):
        model = LabelModel()
        teacher = copy.deepcopy(model)
        x = torch.zeros(3, 1, 2, 2)
        y = torch.ones(3, dtype=torch.long)
        _, hist = mmu_gen(model, teacher, x, y, x[:1], y[:1]*0,
                          Diffusion(T=10, device='cpu'), warmup_epochs=1,
                          epochs=1, bs=2, gamma=0, lr=.1, dev='cpu', log=lambda _:None)
        self.assertEqual([r['phase'] for r in hist], ['warmup', 'joint'])
        self.assertEqual(hist[0]['redirect_loss'], 0)
        self.assertGreater(hist[1]['redirect_loss'], 0)
        self.assertLess(model.values[0].item(), 5)
        torch.testing.assert_close(model.values[1:], teacher.values[1:])

    def test_saved_non_ema_checkpoint_loads(self):
        model = CondUNet(base=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            torch.save({'base': 8, 'model': model.state_dict()}, path)
            loaded = load_source(path, dev='cpu')
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name])

    def test_partial_batch_not_discarded(self):
        x = torch.arange(3)
        self.assertEqual([len(a) for a, _ in batches(x, x, 2, False)], [2, 1])


if __name__ == '__main__':
    unittest.main()
