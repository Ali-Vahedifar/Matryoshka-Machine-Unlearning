import os
import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import classification as c
import torch,numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
sys.argv=['test'];a=c.arg_parser.parse_args();a.unlearn_epochs=1;a.print_freq=10000;a.mask_ratio=.5
base=c.utils.model_dict['resnet18'](num_classes=10).cuda()
base.normalize=c.utils.NormalizeByChannelMeanStd(mean=[.4914,.4822,.4465],std=[.247,.2435,.2616]).cuda()
ds=CIFAR10(os.environ.get('MMU_DATA','data'),train=True,transform=transforms.ToTensor());ds.data=ds.data[:4];ds.targets=np.array(ds.targets[:4])
data={k:c.loader(copy.deepcopy(ds),True) for k in ['forget','retain','test','val']}
for name in ['FT','RL','GA','wfisher','boundary_expanding','boundary_shrink','FT_l1','RL_proximal']:
 model=copy.deepcopy(base);a.unlearn=name;a.save_dir='/tmp/salun_preflight';Path(a.save_dir).mkdir(exist_ok=True)
 c.unlearn.get_unlearn_method(name)(data,model,torch.nn.CrossEntropyLoss(),a)
 assert all(torch.isfinite(p).all() for p in model.parameters())
 print('PASS',name,flush=True)
import importlib.util
spec=importlib.util.spec_from_file_location('mmu_method',c.ROOT.parents[1]/'mmu/MMU/mmu.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
model=copy.deepcopy(base)
mod.MSCRUB(model,epochs=1,msteps=1,diagnose=False).unlearn(data['forget'],data['retain'])
assert all(torch.isfinite(p).all() for p in model.parameters())
print('PASS MMU',flush=True)
