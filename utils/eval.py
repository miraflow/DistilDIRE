import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

from guided_diffusion.compute_dire_eps import dire, create_argparser,dire_get_first_step_noise

from tqdm.auto import tqdm 

from utils.config import CONFIGCLASS
from utils.utils import to_cuda

from utils.datasets import get_binary_distill_dataloader


def get_val_cfg(cfg: CONFIGCLASS, split="val", copy=True):
    if copy:
        from copy import deepcopy

        val_cfg = deepcopy(cfg)
    else:
        val_cfg = cfg
    val_cfg.dataset_root = os.path.join(val_cfg.dataset_root)
    #val_cfg.datasets = cfg.datasets_test
    
    if split == "train":
        val_cfg.isTrain = True
    elif split == "test" or split == "val":
        val_cfg.isTrain = False
    # val_cfg.aug_resize = False
    # val_cfg.aug_crop = False
    val_cfg.aug_flip = False
    val_cfg.serial_batches = True
    val_cfg.jpg_method = ["pil"]
    # Currently assumes jpg_prob, blur_prob 0 or 1
    if len(val_cfg.blur_sig) == 2:
        b_sig = val_cfg.blur_sig
        val_cfg.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_cfg.jpg_qual) != 1:
        j_qual = val_cfg.jpg_qual
        val_cfg.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_cfg

def validate(model: nn.Module, cfg, adm_model, diffusion, adm_args, rank, distributed=False):
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    
    results = dict()
    
    directory = os.path.join(''.join(cfg.dataset_root), "images", "test", "fakes")

    # 디렉토리만 필터링하여 리스트로 가져오기
    fake_dataset_names = [fake_dataset_name for fake_dataset_name in os.listdir(directory) if os.path.isdir(os.path.join(directory, fake_dataset_name))]
    
    for fake_dataset_name in fake_dataset_names:
        data_loader = get_binary_distill_dataloader(cfg, "test", adm_model, diffusion, adm_args, world_size=1,test_fake_dataset_name=fake_dataset_name, distributed=distributed, device=device)
        
        model.to(device)
        model.eval()
        time = 0
        with torch.no_grad():
            y_true, y_pred = [], []
            for data in tqdm(data_loader, total=len(data_loader)):
                un_norm_img, img, dire, label = data
                un_norm_img = un_norm_img.to(device)
                un_norm_img = (un_norm_img * 2) - 1
                eps = dire_get_first_step_noise(un_norm_img, adm_model, diffusion, adm_args, device)
                eps = TF.resize(eps, (224, 224))
                
                img = img.to(device)
                dire = dire.to(device)
                label = label.to(device).float()
                eps = eps.to(device) if eps is not None else None
                st = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                st.record()
                output = model(img, dire, eps)
                # output = model(dire)
                end.record()
                torch.cuda.synchronize()
                time += (st.elapsed_time(end) / 1000)
                predict = output['df_probablity_logit'].sigmoid()
                # predict = output.sigmoid()
                y_pred.extend(predict.flatten().tolist())
                y_true.extend(label.flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
        f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        results[f"{fake_dataset_name} - ACC"] = acc
        results[f"{fake_dataset_name} - AP"] = ap
        results[f"{fake_dataset_name} - R_ACC"] = r_acc
        results[f"{fake_dataset_name} - F_ACC"] = f_acc
        results[f"{fake_dataset_name} - AVG. TIME"] = time / len(data_loader)
        
    return results
