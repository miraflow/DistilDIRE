import os
import time

import torch
import torch.distributed as dist
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as TF
import gc


import PIL.Image as Image

from tensorboardX import SummaryWriter
from tqdm import tqdm
import gc

from guided_diffusion.compute_dire_eps import dire, create_argparser,dire_get_first_step_noise
from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, dict_parse

from utils.config import cfg
# from utils.datasets import get_binary_distill_dataloader
# from utils.eval import get_val_cfg, validate
#from utils.trainer import Trainer
#from utils.utils import Logger
# distributed sampler load
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset, DataLoader

import os.path as osp


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
    print("Synchronization successful!")
    
    
class ExtractDataset(Dataset):
    def __init__(self, img_paths, device):
        self.img_paths = img_paths
        self.device = device

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = torchvision.transforms.ToTensor()(image)
        image = image.to(self.device) * 2 - 1  # 이미지를 GPU로 이동
        # resize
        image = TF.resize(image, (256, 256))
        return img_path, image

# Define your data loader
def get_data_loader(img_paths, device, batch_size):
    dataset = ExtractDataset(img_paths, device)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

    return dataloader

def save_eps(args):
    idx, (e, img_path) = args
    # /media/changyeon/DIRE/datasets/imagenet/images/train/reals/n01440764/ILSVRC2012_val_00000293.JPEG
    # /media/changyeon/DIRE/datasets/celebahq/images/train/fakes/split1_0.png
    save_path = img_path
    save_path = save_path.replace('images', 'eps')
    save_path = save_path.split('.')[0] + '.pt'
    
    save_dir = osp.dirname(save_path)
    
    if not osp.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    
    # if file exists. skip
    if osp.exists(save_path):
        return

    torch.save(e.detach().cpu(), save_path)
    
    del e

def main(rank, world_size, cfg):
    # mp.set_start_method("spawn")
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
    
    roots =  ['/workspace/datasets/imagenet', '/workspace/datasets/celebahq'] #['/media/changyeon/DIRE/datasets/celebahq',
    phases = ['train', 'test'] # phases = ['train', 'test']
    
    labels = {"reals": 0, "fakes": 1}
    
    img_paths = []

    for root in tqdm(roots):
        for phase in tqdm(phases):
            for label, label_idx in labels.items():
                if phase == 'test' and label == 'fakes':
                    test_directory = os.path.join(root, "images", "test", "fakes")
                    fake_dataset_names = [fake_dataset_name for fake_dataset_name in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory, fake_dataset_name))]
                    folder_paths = [os.path.join(root, 'images', phase, label, test_fake_dataset_name) for test_fake_dataset_name in fake_dataset_names]
                
                elif phase == 'train' and label == 'fakes' and 'imagenet' in root:
                    folder_paths = [os.path.join(root, 'images', phase, label, 'adm')]
                    
                else:
                    folder_paths = [os.path.join(root, 'images', phase, label)]
                    
                for folder_path in folder_paths:
                    if (phase == 'train' and 'imagenet' in root) or (phase == 'test' and 'imagenet' in root): # phase == 'train' and 
                        # 'adm' 폴더 안에 있는 서브폴더 리스트 가져오기
                        subfolder_list = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                        
                        # 각 서브폴더에 있는 이미지들을 가져와서 데이터에 추가
                        for subfolder in subfolder_list:
                            subfolder_path = os.path.join(folder_path, subfolder)
                    
                            image_files = os.listdir(subfolder_path)
                            
                            for image_file in image_files:
                                ext = image_file.split('.')[-1].lower()
                                if ext=='jpg' or ext=='png' or ext=='jpeg':
                                    img_path = os.path.join(subfolder_path, image_file)
                                    img_paths.append(img_path)
                            
                    
                    else:
                        image_files = os.listdir(folder_path)
                        
                        for image_file in image_files:
                            ext = image_file.split('.')[-1].lower()
                            if ext=='jpg' or ext=='png' or ext=='jpeg':
                                img_path = os.path.join(folder_path, image_file)
                                img_paths.append((img_path))
    
    # Distribute the dataset among different processes
    # img_paths = img_paths[rank::world_size]
    
    batch_size = 32  # Adjust batch size for distributed data parallelism
                                      
    dataloader = get_data_loader(img_paths, device, batch_size)

    for batch_img_paths, batch_images in tqdm(dataloader):
        with torch.no_grad():
            eps = dire_get_first_step_noise(batch_images, adm_model, diffusion, adm_args, rank)
            # resize (224, 224)
            eps = TF.resize(eps, (224, 224))
            
            for i, (e, img_path) in enumerate(zip(eps, batch_img_paths)):
                save_eps((i, (e, img_path)))    
                
            del eps
            gc.collect()
            
            torch.cuda.empty_cache()
            

    
if __name__ == "__main__":
    n_gpu = int(os.environ.get("WORLD_SIZE", 1))
    distributed = n_gpu > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl")
        synchronize()
    
    main(local_rank, n_gpu, cfg)