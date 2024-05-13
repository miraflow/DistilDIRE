import os
import time

import torch
import torch.distributed as dist
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as TF
import torch.multiprocessing as mp


import PIL.Image as Image

from tensorboardX import SummaryWriter
from tqdm import tqdm
import gc

from guided_diffusion.compute_dire import  create_argparser,dire_get_first_step_noise
from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, dict_parse

from utils.datasets import get_binary_distill_dataloader
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger

from torch.utils.data import Dataset, DataLoader

import os.path as osp
from glob import glob


class TMDistilDireDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.__fake_dire_paths = glob(osp.join(root, 'fakes/DIRE', '*.png'))
        # (imgs, dire)
        self.fake_img_paths = list(map(lambda x: (x.replace('DIRE', 'images').replace('_dire_', '_imgs_'), x, True), self.__fake_dire_paths))

        self.__real_dire_paths = glob(osp.join(root, 'reals/DIRE', '*.png'))
        self.real_img_paths = list(map(lambda x: (x.replace('DIRE', 'images').replace('_dire_', '_imgs_'), x, False), self.__real_dire_paths))
        
        self.img_paths = self.fake_img_paths + self.real_img_paths
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, dire_path, isfake = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = TF.to_tensor(img)*2 - 1
        
        dire = Image.open(dire_path).convert('RGB')
        dire = TF.to_tensor(dire)*2 - 1
        return img, dire, img_path, dire_path, isfake
    


def main():
    # Set device for this process
    device = torch.device("cuda")

    adm_args = create_argparser()
    adm_args['timestep_respacing'] = 'ddim15'
    adm_model, diffusion = create_model_and_diffusion(**dict_parse(adm_args, model_and_diffusion_defaults().keys()))
    adm_model.load_state_dict(torch.load(adm_args['model_path'], map_location="cpu"))
    adm_model.to(device)

    if adm_args['use_fp16']:
        adm_model.convert_to_fp16()
    adm_model.eval()

    dataloader = DataLoader(TMDistilDireDataset("/workspace/truemedia-dire"), 
                            batch_size=32, 
                            shuffle=False, num_workers=2)
    

    for img_batch, dire_batch, img_pathes, dire_pathes, isfakes in tqdm(dataloader):
        with torch.no_grad():
            eps = dire_get_first_step_noise(img_batch, adm_model, diffusion, adm_args, device)
            # center crop (224, 224)
            eps = TF.center_crop(eps, (224, 224)).detach().cpu()
            img = TF.center_crop(img_batch, (224, 224)).detach().cpu()
            dire = TF.center_crop(dire_batch, (224, 224)).detach().cpu()
            
            for i in range(len(img_batch)):
                basename = osp.basename(img_pathes[i])
                isfake = isfakes[i]
                
                img_path = osp.join("/workspace/truemedia-eval-train/images/fakes", basename) if isfake else osp.join("/workspace/truemedia-eval-train/images/reals", basename)
                dire_path = img_path.replace('/images/', '/dire/')
                eps_path = img_path.replace('/images/', '/eps/').split('.')[0] + '.pt'
                
                if not osp.exists(img_path):
                    torchvision.utils.save_image(img[i], img_path)
                if not osp.exists(dire_path):
                    torchvision.utils.save_image(dire[i], dire_path)
                if not osp.exists(eps_path):
                    torch.save(eps[i], eps_path)

    
if __name__ == "__main__":
   main()