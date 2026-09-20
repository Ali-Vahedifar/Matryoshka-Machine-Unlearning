import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from Machine_Unlearning_baselines.common import clone_frozen, kd_kl


def linear_head(model: nn.Module) -> nn.Linear:
    head = getattr(model, 'fc', None)
    if isinstance(head, nn.Linear):
        return head
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise ValueError('MMU needs a linear classifier head to slice by column')
    return linears[-1]


def default_granularities(feature_dim: int, count: int = 5, floor: int = 8):
    dims = {feature_dim // 2 ** k for k in range(count)}
    return tuple(sorted(dim for dim in dims if dim >= floor)) or (feature_dim,)


def _normalised(values, exponent):
    raw = [float(value) ** (-exponent) for value in values]
    total = sum(raw)
    return [value / total for value in raw]


def forget_weights(granularities, gamma):
    return _normalised(range(1, len(granularities) + 1), gamma)


def retain_weights(granularities, beta):
    widest = float(max(granularities))
    return _normalised([m / widest for m in granularities], beta)


def matryoshka_heads(head: nn.Linear, granularities, device):
    heads = {}
    for m in granularities:
        if m == head.in_features:
            continue
        auxiliary = nn.Linear(m, head.out_features,
                              bias=head.bias is not None).to(device)
        with torch.no_grad():
            auxiliary.weight.copy_(head.weight[:, :m])
            if head.bias is not None:
                auxiliary.bias.copy_(head.bias)
        heads[m] = auxiliary
    return heads


def capture_features(model, head, inputs):
    captured = {}
    handle = head.register_forward_hook(
        lambda module, args, output: captured.__setitem__('features', args[0]))
    try:
        model(inputs)
    finally:
        handle.remove()
    return captured['features']


def read(features, head, m, heads=None):
    if heads is not None and m in heads:
        return heads[m](features[:, :m])
    return F.linear(features[:, :m], head.weight[:, :m], head.bias)


def nested_logits(model, head, inputs, granularities, heads=None):
    features = capture_features(model, head, inputs)
    return [read(features, head, m, heads) for m in granularities]


@torch.no_grad()
def prefix_accuracy(model, loader, granularities=None, device='cuda', heads=None):
    head = linear_head(model)
    granularities = granularities or default_granularities(head.in_features)
    was_training = model.training
    model.eval()
    correct = [0] * len(granularities)
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        for index, logits in enumerate(
                nested_logits(model, head, inputs, granularities, heads)):
            correct[index] += int((logits.argmax(dim=1) == targets).sum())
        total += targets.numel()
    model.train(was_training)
    if total == 0:
        raise ValueError('cannot measure prefix accuracy on an empty loader')
    return {int(m): 100.0 * count / total
            for m, count in zip(granularities, correct)}


class MSCRUB:
    def __init__(self, model, device='cuda', epochs=5, msteps=2, lr=5e-4,
                 temperature=4.0, alpha=1.0, gamma=1.0, momentum=0.9,
                 weight_decay=5e-4, granularities=None, head_mode='independent',
                 weighting='mrl', forget_gamma=0.0, retain_beta=1.0,
                 diagnose=True, max_grad_norm=None, bad_margin=None):
        if msteps > epochs:
            raise ValueError('msteps cannot exceed epochs')
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be positive')
        self.max_grad_norm = max_grad_norm
        if head_mode not in ('independent', 'tied'):
            raise ValueError("head_mode must be 'independent' or 'tied'")
        if weighting not in ('mrl', 'normalised'):
            raise ValueError("weighting must be 'mrl' or 'normalised'")
        self.model = model
        self.device = device
        self.epochs = epochs
        self.msteps = msteps
        self.lr = lr
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.head = linear_head(model)
        self.granularities = tuple(granularities or
                                   default_granularities(self.head.in_features))
        if max(self.granularities) > self.head.in_features:
            raise ValueError('a granularity exceeds the feature dimension')
        self.head_mode = head_mode
        self.weighting = weighting
        self.forget_gamma = forget_gamma
        self.retain_beta = retain_beta
        self.diagnose = diagnose
        self.bad_margin = bad_margin
        self.heads = None
        self.diagnostics = {'granularities': list(self.granularities),
                            'head_mode': head_mode, 'weighting': weighting,
                            'bad_margin': bad_margin}
        self.diagnostic_time = 0.0

    def _scales(self):
        if self.weighting == 'mrl':
            ones = [1.0] * len(self.granularities)
            return ones, ones
        return (forget_weights(self.granularities, self.forget_gamma),
                retain_weights(self.granularities, self.retain_beta))

    def unlearn(self, forget_loader, retain_loader):
        teacher = clone_frozen(self.model)
        teacher_head = linear_head(teacher)
        a, b = self._scales()
        self.diagnostics['forget_scales'] = a
        self.diagnostics['retain_scales'] = b
        if self.head_mode == 'independent':
            self.heads = matryoshka_heads(self.head, self.granularities, self.device)
        self._diagnose('prefix_forget_acc_before', forget_loader)

        parameters = list(self.model.parameters())
        for auxiliary in (self.heads or {}).values():
            parameters += list(auxiliary.parameters())
        optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        for epoch in range(self.epochs):
            if epoch < self.msteps:
                self.model.train()
                for inputs, _ in forget_loader:
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        targets = nested_logits(teacher, teacher_head, inputs,
                                                self.granularities)
                    students = self._student_logits(inputs)
                    optimizer.zero_grad(set_to_none=True)
                    if self.bad_margin is None:
                        loss = -sum(
                            scale * kd_kl(student, target, self.temperature)
                            for scale, student, target in zip(a, students, targets))
                    else:
                        loss = sum(
                            scale * F.relu(self.bad_margin
                                           - kd_kl(student, target, self.temperature))
                            for scale, student, target in zip(a, students, targets))
                    loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
                    optimizer.step()

            self.model.train()
            for inputs, labels in retain_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    targets = nested_logits(teacher, teacher_head, inputs,
                                            self.granularities)
                students = self._student_logits(inputs)
                optimizer.zero_grad(set_to_none=True)
                loss = sum(
                    scale * (self.gamma * F.cross_entropy(student, labels)
                             + self.alpha * kd_kl(student, target,
                                                  self.temperature))
                    for scale, student, target in zip(b, students, targets))
                loss.backward()
                optimizer.step()

        self._diagnose('prefix_forget_acc_after', forget_loader)
        if self.heads:
            self._diagnose('prefix_forget_acc_after_tied', forget_loader, heads=False)
            self.heads = None
        return self.model

    def _student_logits(self, inputs):
        features = capture_features(self.model, self.head, inputs)
        return [read(features, self.head, m, self.heads)
                for m in self.granularities]

    def _diagnose(self, key, loader, heads=True):
        if not self.diagnose:
            return
        started = time.perf_counter()
        self.diagnostics[key] = prefix_accuracy(
            self.model, loader, self.granularities, self.device,
            self.heads if heads else None)
        self.diagnostic_time += time.perf_counter() - started
