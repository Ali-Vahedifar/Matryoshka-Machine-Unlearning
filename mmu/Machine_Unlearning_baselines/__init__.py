from .Baseline import Baseline
from .Retrain import Retrain
from .FineTune import FineTune
from .BadTeacher import BadTeacher
from .Amnesiac import Amnesiac
from .UNSIR import UNSIR
from .SSD import SSD
from .UniCLUN import UniCLUN
from .SCRUB import SCRUB
from .SalUn import SalUn

METHODS = {
    'baseline': Baseline,
    'retrain': Retrain,
    'finetune': FineTune,
    'badteacher': BadTeacher,
    'amnesiac': Amnesiac,
    'unsir': UNSIR,
    'ssd': SSD,
    'uniclun': UniCLUN,
    'scrub': SCRUB,
    'salun': SalUn,
}

__all__ = [
    'Baseline', 'Retrain', 'FineTune', 'BadTeacher', 'Amnesiac', 'UNSIR', 'SSD',
    'UniCLUN', 'SCRUB', 'SalUn', 'METHODS'
]
