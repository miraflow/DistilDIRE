import argparse
import os
import sys
from abc import ABC
from typing import Type


class DefaultConfigs(ABC):
    ####### base setting ######
    gpus = [0]
    seed = 3407
    arch = "resnet50"
    datasets = [""]
    datasets_test = [""]
    class_bal = False
    batch_size = 256
    val_every = 1
    loadSize = 256
    cropSize = 224
    epoch = "latest"
    num_workers = 2
    isTrain = True

    ####### train setting ######
    warmup = False
    warmup_epoch = 3
    earlystop = True
    earlystop_epoch = 5
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 20
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    nepoch = 20
    beta1 = 0.9
    lr = 0.00001
    init_type = "normal"
    init_gain = 0.02
    pretrained = True

    # paths information
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_root = os.path.join(root_dir, "datasets")
    dataset_test_root = os.path.join(root_dir, "datasets")
    exp_root = os.path.join(root_dir, "experiments")
    _exp_name = ""
    exp_dir = ""
    ckpt_dir = ""
    logs_path = ""
    ckpt_path = ""
    pretrained_weights = ""

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter
    def exp_name(self, value: str):
        self._exp_name = value
        self.exp_dir: str = os.path.join(self.exp_root, self.exp_name)
        self.ckpt_dir: str = os.path.join(self.exp_dir, "ckpt")
        self.logs_path: str = os.path.join(self.exp_dir, "logs.txt")
        if self.isTrain:
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def to_dict(self):
        dic = {}
        for fieldkey in dir(self):
            fieldvalue = getattr(self, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvalue) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvalue
        return dic


def args_list2dict(arg_list: list):
    assert len(arg_list) % 2 == 0, f"Override list has odd length: {arg_list}; it must be a list of pairs"
    return dict(zip(arg_list[::2], arg_list[1::2]))


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "yes", "on", "y", "t", "1"):
        return True
    elif v.lower() in ("false", "no", "off", "n", "f", "0"):
        return False
    else:
        return bool(v)


def str2list(v: str, element_type=None) -> list:
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


CONFIGCLASS = Type[DefaultConfigs]

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", default="0", type=str)
parser.add_argument("--batch", default=256, type=int)
parser.add_argument("--epoch", default="100", type=int)
parser.add_argument("--exp_name", default="", type=str)
parser.add_argument("--datasets", default="", type=str)
parser.add_argument("--datasets_test", default="", type=str)
parser.add_argument("--pretrained_weights", default="", type=str)
parser.add_argument("--lr", default=0.00001, type=float)
parser.add_argument("--test", default=False, type=str2bool)
parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
args = parser.parse_args()

if os.path.exists(os.path.join(DefaultConfigs.exp_root, args.exp_name, "config.py")):
    sys.path.insert(0, os.path.join(DefaultConfigs.exp_root, args.exp_name))
    from config import cfg

    cfg: CONFIGCLASS
else:
    cfg = DefaultConfigs()

if args.opts:
    opts = args_list2dict(args.opts)
    for k, v in opts.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unrecognized option: {k}")
        original_type = type(getattr(cfg, k))
        if original_type == bool:
            setattr(cfg, k, str2bool(v))
        elif original_type in (list, tuple, set):
            setattr(cfg, k, str2list(v, type(getattr(cfg, k)[0])))
        else:
            setattr(cfg, k, original_type(v))

#cfg.gpus: list = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus#", ".join([str(gpu) for gpu in cfg.gpus])
if args.test:
    cfg.isTrain = False
cfg.exp_name = args.exp_name
cfg.batch_size = args.batch
cfg.datasets = args.datasets
cfg.datasets_test = args.datasets_test if args.datasets_test else args.datasets
cfg.pretrained_weights = args.pretrained_weights
cfg.lr = args.lr
cfg.nepoch = args.epoch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cfg.dataset_root = os.path.join(root_dir, 'datasets', cfg.datasets)
cfg.dataset_test_root = os.path.join(root_dir, 'datasets', cfg.datasets_test)

# if isinstance(cfg.datasets, str):
#     cfg.datasets = cfg.datasets.split(",")
