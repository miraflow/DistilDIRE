import wandb
from utils.config import cfg
import torch


    
def main(run, cfg):
    from utils.trainer import Trainer
    from torch.utils.data import DataLoader
    from dataset import TMDistilDireDataset
    # Set device for this process
    device = torch.device("cuda")
    dataloader = DataLoader(TMDistilDireDataset("/workspace/truemedia-eval-train"), 
                            batch_size=128, 
                            shuffle=True, num_workers=2)
    trainer = Trainer(cfg, dataloader, dataloader, run, 0, False, 1)
    trainer.load_networks("/workspace/DIRE/model_epoch_50.pth")
    trainer.train()

if __name__ == "__main__":
    print("Cuda is available: ", torch.cuda.is_available())

    run = wandb.init(project=f'dire-distill-truemedia', config=cfg) 
    main(run, cfg)
    