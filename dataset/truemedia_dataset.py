from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from glob import glob
import os.path as osp
from PIL import Image


class TMDistilDireDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.__fake_img_paths = glob(osp.join(root, 'images/fakes/', '*.png'))
        # (imgs, dire, eps, isfake)
        self.fake_paths = list(map(lambda x: (x, x.replace('/images/', '/dire/'), x.replace('/images/', '/eps/').split('.')[0]+'.pt', True), self.__fake_img_paths))

        self.__real_img_paths = glob(osp.join(root, 'images/reals/', '*.png'))
        # (imgs, dire, eps, isfake)
        self.real_paths = list(map(lambda x: (x, x.replace('/images/', '/dire/'), x.replace('/images/', '/eps/').split('.')[0]+'.pt', False), self.__real_img_paths))

        self.img_paths = self.fake_paths + self.real_paths
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, dire_path, eps_path, isfake = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = TF.to_tensor(img)*2 - 1
        
        dire = Image.open(dire_path).convert('RGB')
        dire = TF.to_tensor(dire)*2 - 1
        
        eps = torch.load(eps_path, map_location='cpu')
        return (img, dire, eps, isfake), (img_path, dire_path, eps_path)