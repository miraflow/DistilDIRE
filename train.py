import wandb
from utils.config import cfg
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os 
import torch


def main(run, cfg):
    from utils.trainer import Trainer
    from torch.utils.data import DataLoader
    from dataset import TMDistilDireDataset
    
    dataloader = DataLoader(TMDistilDireDataset(cfg.dataset_root), 
                            batch_size=cfg.batch_size, 
                            shuffle=True, num_workers=2)
    trainer = Trainer(cfg, dataloader, dataloader, run, local_rank, True, world_size)
    if cfg.pretrained_weights:
        trainer.load_networks(cfg.pretrained_weights)
    trainer.train()

if __name__ == "__main__":
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK']) 
    world_size = int(os.environ['WORLD_SIZE'])
    run = None
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        run = wandb.init(project=f'dire-distill-truemedia', config=cfg, dir=cfg.exp_dir) 
    main(run, cfg)
    