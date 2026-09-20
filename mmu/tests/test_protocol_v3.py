from types import SimpleNamespace

import numpy as np

from scripts.benchmark_unlearning import Benchmark, finite_json
from scripts.ulira import attack_metrics, balanced_masks, likelihood_ratio


def _bench(selection='utility', tolerance=2.0):
    stub = Benchmark.__new__(Benchmark)
    stub.args = SimpleNamespace(selection=selection, utility_tolerance=tolerance)
    return stub


def _row(forget, retain, test, mia=50.0):
    return {'valid': True, 'gap_to_retrain': 0.0,
            'selection_metrics': {'forget_acc': forget, 'retain_acc': retain,
                                  'test_acc': test, 'mia': mia}}


GOLD = {'forget_acc': 0.0, 'retain_acc': 90.0, 'test_acc': 85.0, 'mia': 50.0}


def test_utility_selection_prefers_forgetting_inside_the_budget():
    rows = [_row(0.0, 80.0, 75.0),
            _row(5.0, 89.0, 84.5),
            _row(2.0, 88.5, 84.0)]
    chosen = _bench().select(rows, GOLD)
    assert chosen is rows[2] and chosen['feasible']
    assert rows[0]['feasible'] is False


def test_utility_selection_falls_back_and_flags_when_nothing_is_feasible():
    rows = [_row(0.0, 60.0, 55.0), _row(0.0, 70.0, 65.0)]
    chosen = _bench().select(rows, GOLD)
    assert chosen is rows[1] and chosen['feasible'] is False


def test_gap_selection_is_the_legacy_rule():
    rows = [_row(0.0, 80.0, 75.0), _row(5.0, 89.0, 84.5)]
    rows[0]['gap_to_retrain'], rows[1]['gap_to_retrain'] = 3.3, 2.0
    assert _bench('gap').select(rows, GOLD) is rows[1]


def test_finite_json_replaces_nan_and_inf_only():
    payload = {'a': float('nan'), 'b': [1.0, float('inf'), {'c': -float('inf')}], 'd': 2}
    assert finite_json(payload) == {'a': None, 'b': [1.0, None, {'c': None}], 'd': 2}


def test_ulira_separates_in_from_out_and_is_chance_on_retrain():
    rng = np.random.default_rng(0)
    K, n = 16, 200
    in_mask = rng.random((K, n)) < 0.5
    mean_in, mean_out = 3.0, 0.0
    shadows = np.where(in_mask, mean_in, mean_out) + rng.normal(0, 1, (K, n))
    unlearned_target = mean_in + rng.normal(0, 1, n)
    retrained_target = mean_out + rng.normal(0, 1, n)
    pos = likelihood_ratio(shadows, in_mask, unlearned_target)
    neg = likelihood_ratio(shadows, in_mask, retrained_target)
    assert attack_metrics(pos, neg)['auc'] > 95
    same = likelihood_ratio(shadows, in_mask, mean_out + rng.normal(0, 1, n))
    assert abs(attack_metrics(same, neg)['auc'] - 50) < 8


def test_balanced_masks_give_every_example_half_in_half_out():
    masks = balanced_masks(16, 500, seed=3)
    assert masks.shape == (16, 500) and (masks.sum(0) == 8).all()
    assert np.array_equal(masks, balanced_masks(16, 500, seed=3))


def _granularity_stub(backbone, num_classes, floor, hidden=64):
    from core.models.backbones import get_backbone
    stub = Benchmark.__new__(Benchmark)
    stub.args = SimpleNamespace(mmu_granularity_floor=floor)
    kwargs = {'hidden_size': hidden} if backbone == 'fpecnn' else {'small_input': True}
    stub.factory = lambda: get_backbone(backbone, num_classes=num_classes, **kwargs)
    return stub


def test_default_floor_preserves_the_cifar10_granularities():
    assert _granularity_stub('resnet18', 10, 8)._mmu_granularities() == (32, 64, 128, 256, 512)
    assert _granularity_stub('fpecnn', 10, 8)._mmu_granularities() == (8, 16, 32, 64)


def test_class_count_floor_drops_rank_deficient_prefixes():
    assert _granularity_stub('resnet18', 100, 100)._mmu_granularities() == (128, 256, 512)
    assert _granularity_stub('resnet18', 200, 200)._mmu_granularities() == (256, 512)
    assert _granularity_stub('fpecnn', 100, 100, hidden=512)._mmu_granularities() == (128, 256, 512)


def test_a_head_narrower_than_the_label_space_is_refused():
    import pytest
    with pytest.raises(RuntimeError, match='no MMU nesting width'):
        _granularity_stub('fpecnn', 100, 100, hidden=64)._mmu_granularities()


def test_rti_label_space_and_forget_selection():
    from core.datasets.cifar100_coarse import CoarseLabelCIFAR100

    class _Base:
        data = np.zeros((6, 4, 4, 3), dtype=np.uint8)
        targets = [0, 1, 5, 6, 10, 11]
        transform = None

    fine_to_coarse = [0] * 100
    for fine, coarse in ((0, 3), (1, 3), (5, 7), (6, 7), (10, 9), (11, 9)):
        fine_to_coarse[fine] = coarse

    coarse_view = CoarseLabelCIFAR100(_Base(), fine_to_coarse, expose_coarse_targets=True)
    fine_view = CoarseLabelCIFAR100(_Base(), fine_to_coarse)
    assert coarse_view.targets == [3, 3, 7, 7, 9, 9]
    assert fine_view.targets == [0, 1, 5, 6, 10, 11]
    assert [coarse_view[i][1] for i in range(6)] == [3, 3, 7, 7, 9, 9]
    assert [fine_view[i][1] for i in range(6)] == [3, 3, 7, 7, 9, 9]
