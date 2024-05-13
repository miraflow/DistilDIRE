import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
import wandb
import gc
from utils.utils import get_network
from guided_diffusion.compute_dire import dire, create_argparser
from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, dict_parse

from utils.config import cfg
from utils.datasets import get_binary_distill_dataloader
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger



def synchronize():
    if not dist.is_available():
        print("torch.distributed is not available")
        return

    if not dist.is_initialized():
        print("torch.distributed is not initialized")
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def main(rank, world_size, run, cfg):
    # Set device for this process
    device = torch.device("cuda")

    val_cfg = get_val_cfg(cfg, split="test", copy=True)
    cfg.dataset_root = os.path.join(cfg.dataset_root)
    
    val_loader = get_binary_distill_dataloader(val_cfg, "test", world_size > 1)
    trainer = Trainer(cfg, None, val_loader, run, rank, world_size > 1, world_size)
    trainer.load_networks(50)
    trainer.validate(True)
    
if __name__ == "__main__":
    n_gpu = int(os.environ.get("WORLD_SIZE", 1))
    distributed = n_gpu > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl")
        synchronize()
    
    run = None
    if local_rank == 0:
        run = wandb.init(project=f'dire-distill-cvprw-test', config=cfg) 
    main(local_rank, n_gpu, run, cfg)
    