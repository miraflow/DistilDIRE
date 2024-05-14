import wandb
from utils.config import cfg

    
def main(run, cfg):
    from utils.trainer import Trainer
    from torch.utils.data import DataLoader
    from dataset import TMDistilDireDataset
    
    dataloader = DataLoader(TMDistilDireDataset(cfg.dataset_root), 
                            batch_size=cfg.batch_size, 
                            shuffle=True, num_workers=2)
    trainer = Trainer(cfg, dataloader, dataloader, run, 0, False, 1)
    if cfg.pretrained_weights:
        trainer.load_networks(cfg.pretrained_weights)
    trainer.train()

if __name__ == "__main__":

    run = wandb.init(project=f'dire-distill-truemedia', config=cfg, dir=cfg.exp_dir) 
    main(run, cfg)
    