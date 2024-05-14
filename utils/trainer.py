import os
import torch
import torch.nn as nn

import torchvision.models as TVM
from collections import OrderedDict
from sklearn.metrics import accuracy_score, average_precision_score
    
from tqdm.auto import tqdm
import numpy as np

from utils.config import CONFIGCLASS
from networks.distill_model import DistilDIRE
from utils.warmup import GradualWarmupScheduler


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

    def __init__(self, cfg: CONFIGCLASS, train_loader, val_loader, run, rank=0, distributed=True, world_size=1):
        super().__init__(cfg)
        self.arch = cfg.arch
        self.rank = rank
        self.device = torch.device("cuda") 
        self.distributed = distributed
        self.world_size = world_size
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.val_every = cfg.val_every
        self.cur_epoch = 0
        
        # wandb logger (pass if None)
        self.run = run
        
        self.student = DistilDIRE(self.device).to(self.device)
        __backbone = TVM.resnet50(weights=TVM.ResNet50_Weights.DEFAULT)
        self.teacher = nn.Sequential(OrderedDict([*(list(__backbone.named_children())[:-2])])) # drop last layer which is classifier
        self.teacher.eval().to(self.device)
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.kd_criterion = nn.MSELoss(reduction='mean')
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
        
        if cfg.continue_train:
            self.load_networks(cfg.epoch)
        
        # AMP
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True


    def set_input(self, input):
        img, dire, eps, label = input # if len(input) == 3 else (input[0], input[1], {})
            
        self.input = img.to(self.device)
        self.dire = dire.to(self.device)
        self.eps = eps.to(self.device)
        self.label = label.to(self.device).float()
       
   
    def forward(self):
        self.output = self.student(self.input, self.eps)
        with torch.no_grad():
            self.teacher_feature = self.teacher(self.dire)


    def get_loss(self):
        loss1 = self.cls_criterion(self.output['logit'].squeeze(), self.label) 
        loss2 = self.kd_criterion(self.output['feature'], self.teacher_feature)
        return loss1 + loss2 * 0.5


    @torch.cuda.amp.autocast()
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss = self.get_loss()
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
    
    def load_networks(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        model_state_dict = state_dict["model"]
        optimizer_state_dict = state_dict["optimizer"]
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        optimizer_state_dict = {k.replace("module.", ""): v for k, v in optimizer_state_dict.items()}

        self.student.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        
        print(f"Model loaded from {model_path}")
        return True 
    
    @torch.no_grad()
    def validate(self, gather=False):
        self.student.eval()
        y_pred = []
        y_true = []
        N_FAKE, N_REAL = 0, 0
        for data, path in tqdm(self.val_loader, desc=f"Validation after {self.cur_epoch} epoch..."):
            self.set_input(data)
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
        
        
    def train(self):
        for epoch in range(self.cfg.nepoch):
            self.cur_epoch += 1
            if self.run:
                self.run.log({"epoch": epoch})
            if epoch % self.val_every == 0 and epoch != 0:
                self.validate()
            
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