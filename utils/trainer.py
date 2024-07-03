import os
import torch
import torch.nn as nn

import torchvision.models as TVM
from collections import OrderedDict
from sklearn.metrics import accuracy_score, average_precision_score
import torch.distributed as dist
from tqdm.auto import tqdm
import numpy as np

from utils.config import CONFIGCLASS
from networks.distill_model import DistilDIRE, DIRE, DistilDIREOnlyEPS
from utils.warmup import GradualWarmupScheduler
import os.path as osp
from guided_diffusion.compute_dire_eps import dire_get_first_step_noise, create_dicts_for_static_init
from guided_diffusion.guided_diffusion.script_util import (
                model_and_diffusion_defaults,
                create_model_and_diffusion,
                add_dict_to_argparser,
                dict_parse,
                args_to_dict,
)

class BaseModel(nn.Module):
    def __init__(self, cfg: CONFIGCLASS):
        super().__init__()
        self.cfg = cfg
        self.total_steps = 0
        self.isTrain = cfg.isTrain
        self.save_dir = cfg.ckpt_dir
        self.nepoch = cfg.nepoch
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.student: nn.Module
        self.optimizer: torch.optim.Optimizer

    def save_networks(self, epoch: int):
        save_filename = f"model_epoch_{epoch}.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.student.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)

    # load models from the disk
    def load_networks(self, epoch: int):
        load_filename = f"model_epoch_{epoch}.pth"
        load_path = os.path.join(self.save_dir, load_filename)

        print(f"loading the model from {load_path}")
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=self.device)["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        self.student.load_state_dict(state_dict)
        self.total_steps = state_dict["total_steps"]

        if self.isTrain and not self.cfg.new_optim:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            # move optimizer state to GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.lr

    def eval(self):
        self.student.eval()

    def test(self):
        with torch.no_grad():
            self.forward()


class Trainer(BaseModel):
    def name(self):
        return "DistilDIRE Trainer"

    def __init__(self, cfg: CONFIGCLASS, train_loader, val_loader, run, rank=0, distributed=True, world_size=1, kd=True):
        super().__init__(cfg)
        self.arch = cfg.arch
        self.reproduce_dire = cfg.reproduce_dire
        self.only_eps = cfg.only_eps
        self.only_img = cfg.only_img
        self.test_name = osp.basename(cfg.dataset_test_root)
        self.rank = rank
        self.device = torch.device(f"cuda") 
        self.distributed = distributed
        self.world_size = world_size
        self.kd = kd
        self.kd_weight = cfg.kd_weight
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.val_every = cfg.val_every
        self.cur_epoch = 0
        
        # wandb logger (pass if None)
        self.run = run
        self.adm = None
        if self.reproduce_dire:
            self.student = DIRE(self.device).to(self.device)
        else: 
            if self.only_eps or self.only_img:
                self.student = DistilDIREOnlyEPS(self.device).to(self.device)
                if self.only_img:
                    adm_args = create_dicts_for_static_init()
                    adm_args['timestep_respacing'] = 'ddim20'
                    adm_model, diffusion = create_model_and_diffusion(**dict_parse(adm_args, model_and_diffusion_defaults().keys()))
                    adm_model.load_state_dict(torch.load(adm_args['model_path'], map_location="cpu"))
                    print("ADM model loaded...")
                    self.adm = adm_model
                    self.adm.convert_to_fp16()
                    self.adm.to(self.device)
                    self.adm.eval()
                    self.diffusion = diffusion
                    self.adm_args = adm_args
            else:
                self.student = DistilDIRE(self.device).to(self.device)
            # self.student.convert_to_fp16_student()
            __backbone = TVM.resnet50(weights=TVM.ResNet50_Weights.DEFAULT)
            self.teacher = nn.Sequential(OrderedDict([*(list(__backbone.named_children())[:-2])])) # drop last layer which is classifier
            self.teacher.eval().to(self.device)
            # Freeze teacher model
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.kd_criterion = nn.MSELoss(reduction='mean')
        
        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        # initialize optimizers
        if cfg.optim == "adam":
            self.optimizer = torch.optim.Adam(self.student.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        elif cfg.optim == "sgd":
            self.optimizer = torch.optim.SGD(self.student.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError("optim should be [adam, sgd]")
        
        if self.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            self.student = DDP(self.student, device_ids=[rank], output_device=rank)
        
        if cfg.warmup:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, cfg.nepoch - cfg.warmup_epoch, eta_min=1e-6
            )
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, multiplier=1, total_epoch=cfg.warmup_epoch, after_scheduler=scheduler_cosine
            )
            self.scheduler.step()        

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True


    def set_input(self, input, istrain=True):
        if self.reproduce_dire:
            dire, label = input
            self.dire = dire.to(self.device)
            self.label = label.to(self.device).float()
            
        else:
            img, dire, eps, label = input # if len(input) == 3 else (input[0], input[1], {})
            H, W = img.shape[-2:]
            B = img.shape[0]

            # only-img
            if self.only_img:
                img = img.to(self.device)
                # calc eps from img
                eps = dire_get_first_step_noise(img, self.adm, self.diffusion, self.adm_args, self.device)

            # cutmix
            if torch.rand(1) < 0.3 and istrain:
                c_lambda = torch.rand(1)
                r_x = torch.randint(0, W, (1,))
                r_y = torch.randint(0, H, (1,))
                r_w = int(torch.sqrt(1-c_lambda)*W)
                r_h = int(torch.sqrt(1-c_lambda)*H)

                img[:, :, r_y:r_y+r_h, r_x:r_x+r_w] = img[0:1, :, r_y:r_y+r_h, r_x:r_x+r_w].repeat(B, 1, 1, 1)
                dire[:, :, r_y:r_y+r_h, r_x:r_x+r_w] = dire[0:1, :, r_y:r_y+r_h, r_x:r_x+r_w].repeat(B, 1, 1, 1)
                eps[:, :, r_y:r_y+r_h, r_x:r_x+r_w] = eps[0:1, :, r_y:r_y+r_h, r_x:r_x+r_w].repeat(B, 1, 1, 1)
                label = c_lambda * label + (1-c_lambda) * label[0:1]
            self.input = img.to(self.device)
            self.dire = dire.to(self.device)
            self.eps = eps.to(self.device)
            self.label = label.to(self.device).float()
       
   
    def forward(self):
        if self.reproduce_dire:
            self.output = self.student(self.dire)
        else:
            if self.only_eps or self.only_img:
                self.output = self.student(self.eps)
            else:
                self.output = self.student(self.input, self.eps)
            with torch.no_grad():
                self.teacher_feature = self.teacher(self.dire)


    def get_loss(self, kd=True):
        loss = self.cls_criterion(self.output['logit'].squeeze(), self.label) 
        if kd and (not self.reproduce_dire):
            loss2 = self.kd_criterion(self.output['feature'], self.teacher_feature)
            loss = loss + loss2 * self.kd_weight
        return loss


    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss = self.get_loss(self.kd)
        self.loss.backward()
        self.optimizer.step()

    
    
    def load_networks(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        model_state_dict = state_dict["model"]
        optimizer_state_dict = state_dict["optimizer"]
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        optimizer_state_dict = {k.replace("module.", ""): v for k, v in optimizer_state_dict.items()}

        if self.distributed:
            self.student.module.load_state_dict(model_state_dict)
        else:
            self.student.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        
        print(f"Model loaded from {model_path}")
        return True 
    
    @torch.no_grad()
    def validate(self, gather=False, save=False, save_name=""):
        self.student.eval()
        y_pred = []
        y_true = []
        N_FAKE, N_REAL = 0, 0
        for data, path in tqdm(self.val_loader, desc=f"Validation after {self.cur_epoch} epoch..."):
            self.set_input(data, istrain=False)
            self.forward()
            pred = self.output['logit'].sigmoid()
            
            if gather:
                try:
                    dist
                except:
                    import torch.distributed as dist
                pred_gather = [pred for _ in range(self.world_size)]
                label_gather = [self.label for _ in range(self.world_size)]
                dist.all_gather(pred_gather, pred)
                dist.all_gather(label_gather, self.label)
            else:
                pred_gather = [pred]
                label_gather = [self.label]
            N_FAKE += sum([(label == 1).sum().item() for label in label_gather])
            N_REAL += sum([(label == 0).sum().item() for label in label_gather])
            
            y_pred.extend(torch.cat(pred_gather).flatten().detach().cpu().tolist())
            y_true.extend(torch.cat(label_gather).flatten().detach().cpu().tolist())
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        acc = accuracy_score(y_true, y_pred > 0.5)
        ap = average_precision_score(y_true, y_pred)
        if self.run:
            self.run.log({"val_acc": acc, "val_ap": ap})
            self.run.log({"N_FAKE": N_FAKE, "N_REAL": N_REAL})
        print(f"Validation: acc: {acc}, ap: {ap}")
        print(f"N_FAKE: {N_FAKE}, N_REAL: {N_REAL}")
        if save:
            with open(save_name, "w") as f:
                f.write(f"Validation: acc: {acc}, ap: {ap}")
                f.write(f"N_FAKE: {N_FAKE}, N_REAL: {N_REAL}")
        
        
    def train(self):
        for epoch in range(self.cfg.nepoch):
            if self.run:
                self.run.log({"epoch": epoch})
            if epoch % self.val_every == 0 and epoch != 0:
                self.validate(gather=self.distributed)
            
            self.student.train()
            for data_and_paths in tqdm(self.train_loader, desc=f"Trainig {epoch} epoch..."):
                data, path = data_and_paths
                self.total_steps += 1
                self.set_input(data)
                self.optimize_parameters()
                if self.total_steps % 100 == 0 and self.run:
                    print(f"step: {self.total_steps}, loss: {self.loss}")
                    self.run.log({"loss": self.loss, "step": self.total_steps})
            
            if self.run:
                self.save_networks(epoch)
                
            if self.cfg.warmup:
                self.scheduler.step()
            if self.distributed:
                dist.barrier()
            self.cur_epoch = epoch