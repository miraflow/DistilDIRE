import wandb
from utils.config import cfg

    
def main(run, cfg):
    from utils.trainer import Trainer
    from torch.utils.data import DataLoader
    from dataset import TMDistilDireDataset
    
    dataloader = DataLoader(TMDistilDireDataset(cfg.dataset_root), 
                            batch_size=1, 
                            shuffle=True, num_workers=2)
    trainer = Trainer(cfg, dataloader, dataloader, run, 0, False, 1)
    assert len(cfg.pretrained_weights) != 0, "Give proper checkpoint path"
    trainer.load_networks(cfg.pretrained_weights)
    trainer.validate(False)

if __name__ == "__main__":
    main(None, cfg)
    