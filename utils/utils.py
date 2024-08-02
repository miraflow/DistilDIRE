import argparse
import warnings
import numpy as np
import torch


warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def str2bool(v: str, strict=True) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ("true", "yes", "on" "t", "y", "1"):
            return True
        elif v.lower() in ("false", "no", "off", "f", "n", "0"):
            return False
    if strict:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")
    else:
        return True


def to_cuda(data, device="cuda", exclude_keys: "list[str]" = None):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, (tuple, list, set)):
        data = [to_cuda(b, device) for b in data]
    elif isinstance(data, dict):
        if exclude_keys is None:
            exclude_keys = []
        for k in data.keys():
            if k not in exclude_keys:
                data[k] = to_cuda(data[k], device)
    else:
        # raise TypeError(f"Unsupported type: {type(data)}")
        data = data
    return data


def pad_img_to_square(img: np.ndarray):
    H, W = img.shape[:2]
    if H != W:
        new_size = max(H, W)
        img = np.pad(img, ((0, new_size - H), (0, new_size - W), (0, 0)), mode="constant")
        assert img.shape[0] == img.shape[1] == new_size
    return img
