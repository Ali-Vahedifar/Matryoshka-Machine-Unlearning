import os
import copy
import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import classification as C
from gradient_pass_diagnostic import M
from joint_target_one_seed import ROOT, take


def main():
    out = ROOT/'old_recipe_one_seed'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    sys.argv = ['old-recipe']
    args = C.arg_parser.parse_args()
    args.data = os.environ.get('MMU_DATA','data')
    args.seed = args.train_seed = 2
    original = C.utils.cifar10_dataloaders
    def same_split(*v, **kw):
        kw.setdefault('seed', 2)
        return original(*v, **kw)
    C.utils.cifar10_dataloaders = same_split
    model, tr, va, te, _ = C.utils.setup_model_dataset(args)
    model.cuda()
    base = ROOT/'classification/seed2'
    model.load_state_dict(torch.load(base/'source.pt', weights_only=False)['state_dict'])
    split = np.load(base/'forget10/split.npz')
    forget = take(tr.dataset, np.sort(split['forget']))
    retain = take(tr.dataset, split['retain'])
    indices = np.random.default_rng(2).choice(len(retain), round(.1*len(retain)), replace=False)
    small = take(retain, indices)
    np.save(out/'retain_subset_indices.npy', indices)
    reference = ROOT.parents[1]/'results/classification_benchmark/cifar10/resnet18_adam_instance/seed42/final.json'
    config = json.loads(reference.read_text())['results']['mmu']['config']
    C.dump(out/'config.json', dict(seed=2, ratio=.1, recipe_source=str(reference),
                                 hyperparameters=config, retain_samples=len(small),
                                 forget_samples=len(forget), batch_size=64,
                                 note='Archived recipe transferred without tuning; same current source, split and full-set evaluator.'))
    C.utils.setup_seed(2)
    method = M.MSCRUB(model, diagnose=False, **config)
    start = time.perf_counter()
    method.unlearn(DataLoader(forget, batch_size=64, shuffle=True, num_workers=0),
                   DataLoader(small, batch_size=64, shuffle=True, num_workers=0))
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    assert all(torch.isfinite(p).all() for p in model.parameters())
    torch.save({'state_dict': model.state_dict(), 'config': config}, out/'model.pt')
    result = C.evaluate(model, retain, forget, te.dataset)
    result.update(seed=2, method='MMU_old_recipe', RTE_min=seconds/60)
    C.dump(out/'results.json', result)
    print('OLD RECIPE TRANSFER COMPLETE', result, flush=True)


if __name__ == '__main__':
    main()
