
import os 
import torch


def main(run, cfg):
    from torch.utils.data.distributed import DistributedSampler
    from utils.trainer import Trainer
    
    if cfg.reproduce_dire:
        dataset = TMDireDataset(cfg.dataset_root)
        val_dataset = TMDireDataset(cfg.dataset_test_root)
    elif cfg.only_eps:
        dataset = TMEPSOnlyDataset(cfg.dataset_root)
        val_dataset = TMEPSOnlyDataset(cfg.dataset_root)
    
    elif cfg.only_img:
        dataset = TMIMGOnlyDataset(cfg.dataset_root, istrain=True)
        val_dataset = TMIMGOnlyDataset(cfg.dataset_test_root, istrain=False)
    else:
        # dataset= TMDistilDireDataset(cfg.dataset_root)
        # val_dataset = TMDistilDireDataset(cfg.dataset_test_root)
        # 2024.7.15 only
        roots = ["/home/ubuntu/y1/DistilDIRE/datasets/truemedia-total", "/home/ubuntu/y1/DistilDIRE/datasets/truemedia-external"]
        eps_roots = ["/home/ubuntu/y1/DistilDIRE/datasets/truemedia-total", "/home/ubuntu/y1/DistilDIRE/datasets/truemedia-external"]
        dataset = JOINEDDistilDireDataset(roots, eps_roots)
        # dataset = TMDistilDireDataset(cfg.dataset_root)
        val_dataset = TMDistilDireDataset(cfg.dataset_test_root)
    sampler = DistributedSampler(dataset)
    val_samlper = DistributedSampler(val_dataset)
    dataloader = DataLoader(dataset, 
                            batch_size=cfg.batch_size, 
                            sampler=sampler,
                            num_workers=4)
    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.batch_size, 
                            sampler=val_samlper,
                            num_workers=4)
    trainer = Trainer(cfg, dataloader, val_loader, run, local_rank, True, world_size, cfg.kd)
    if cfg.pretrained_weights:
        trainer.load_networks(cfg.pretrained_weights)
    trainer.train()


if __name__ == "__main__":
    
    import torch.distributed as dist
    import os 
    import wandb
    
    from torch.utils.data import DataLoader
    from dataset import TMDistilDireDataset, TMDireDataset, TMEPSOnlyDataset, TMIMGOnlyDataset, JOINEDDistilDireDataset
    
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK']) 
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.barrier()
    from utils.config import cfg
    run = None
    if local_rank == 0:
        run = wandb.init(project=f'dire-distill-truemedia', config=cfg, dir=cfg.exp_dir) 
    main(run, cfg)
    

