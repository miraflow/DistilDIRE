from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, CenterCrop
from glob import glob
import os.path as osp
from PIL import Image


class TMDistilDireDataset(Dataset):
    def __init__(self, root, prepared_dire=True):
        self.root = root
        self.__fake_img_paths = glob(osp.join(root, 'images/fakes/', '*.png'))
        self.__real_img_paths = glob(osp.join(root, 'images/reals/', '*.png'))
        self.prepared_dire = prepared_dire
        if self.prepared_dire:
            self.transform = Compose([Resize(224), CenterCrop((224, 224))])
        else:
            self.transform = Compose([Resize(256), CenterCrop((256, 256))])
        

        # (imgs, dire, eps, isfake)
        if prepared_dire:
            self.fake_paths = list(map(lambda x: (x, x.replace('/images/', '/dire/'), x.replace('/images/', '/eps/').split('.')[0]+'.pt', True), self.__fake_img_paths))
            self.real_paths = list(map(lambda x: (x, x.replace('/images/', '/dire/'), x.replace('/images/', '/eps/').split('.')[0]+'.pt', False), self.__real_img_paths))
        else:
            self.fake_paths = list(map(lambda x: (x, "", ""), self.__fake_img_paths))
            self.real_paths = list(map(lambda x: (x, "", ""), self.__real_img_paths))
        self.img_paths = self.fake_paths + self.real_paths
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, dire_path, eps_path, isfake = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = TF.to_tensor(img)*2 - 1
        
        if self.prepared_dire:
            dire = Image.open(dire_path).convert('RGB')
            dire = TF.to_tensor(dire)*2 - 1
            eps = torch.load(eps_path, map_location='cpu')
            assert img.shape[1:] == dire.shape[1:] == eps.shape[1:], f"Shape mismatch: {img.shape[1:]}, {dire.shape[1:]}, {eps.shape[1:]}"
            
        else:
            img = self.transform(img)
            dire = torch.zeros_like(img)
            eps = torch.zeros_like(img)

        return (img, dire, eps, isfake), (img_path, dire_path, eps_path)