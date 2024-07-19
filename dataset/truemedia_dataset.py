from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.io import decode_jpeg, encode_jpeg
from glob import glob
import os.path as osp
from PIL import Image
import random
import os 
TARGET_COMP = 0.1

class TMDistilDireDataset(Dataset):
    def __init__(self, root, prepared_dire=True):
        self.root = root
        self.__fake_img_paths = [p for p in glob(osp.join(root, 'images/fakes/', '*')) if p.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'webp']]
        self.__real_img_paths = [p for p in glob(osp.join(root, 'images/reals/', '*')) if p.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'webp']]
        self.prepared_dire = prepared_dire
        self.transform = Compose([Resize(256, antialias='True'), CenterCrop((256, 256))])
        

        # (imgs, dire, eps, isfake)
        if prepared_dire:
            self.fake_paths = list(map(lambda x: (x, x.replace('/images/', '/dire/'), x.replace('/images/', '/eps/').split('.')[0]+'.pt', True), self.__fake_img_paths))
            self.real_paths = list(map(lambda x: (x, x.replace('/images/', '/dire/'), x.replace('/images/', '/eps/').split('.')[0]+'.pt', False), self.__real_img_paths))
        else:
            self.fake_paths = list(map(lambda x: (x, "", "", True), self.__fake_img_paths))
            self.real_paths = list(map(lambda x: (x, "", "", False), self.__real_img_paths))
        # random.shuffle(self.fake_paths)
        # random.shuffle(self.real_paths)
        self.img_paths = self.fake_paths + self.real_paths
        if prepared_dire:
            img_paths = []
            for img_path, dire_path, eps_path, isfake in self.img_paths:
                if not osp.exists(eps_path) or not osp.exists(img_path) or not osp.exists(dire_path):
                    continue
                try:
                    Image.open(img_path)
                    img_paths.append((img_path, dire_path, eps_path, isfake))
                except:
                    continue
            self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, dire_path, eps_path, isfake = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = TF.to_tensor(img)*2 - 1

        
        if self.prepared_dire:
            img = self.transform(img)
            dire = Image.open(dire_path).convert('RGB')
            dire = TF.to_tensor(dire)*2 - 1
            dire = self.transform(dire)
            eps = torch.load(eps_path, weights_only=True, mmap=True)
            assert img.shape[1:] == dire.shape[1:] == eps.shape[1:], f"Shape mismatch: {img.shape[1:]}, {dire.shape[1:]}, {eps.shape[1:]}"
            
        else:
            img = self.transform(img)
            dire = torch.zeros_like(img)
            eps = torch.zeros_like(img)

        return (img, dire, eps, isfake), (img_path, dire_path, eps_path)


class JOINEDDistilDireDataset(Dataset):
    def __init__(self, roots, eps_roots=None, istrain=True):
        self.roots = roots
        self.__fake_img_paths = []
        self.__real_img_paths = []
        self.istrain = istrain
        for root in roots:
            self.__fake_img_paths += [p for p in glob(osp.join(root, 'images/fakes/', '*')) if p.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'webp']]
            self.__real_img_paths += [p for p in glob(osp.join(root, 'images/reals/', '*')) if p.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'webp']]
        
        self.transform = Compose([Resize(256, antialias='True'), CenterCrop((256, 256))])
        
        if eps_roots is None:
            # (imgs, dire, eps, isfake)
            self.fake_paths = list(map(lambda x: (x, x.replace('/images/', '/dire/'), x.replace('/images/', '/eps/').split('.')[0]+'.pt', True), self.__fake_img_paths))
            self.real_paths = list(map(lambda x: (x, x.replace('/images/', '/dire/'), x.replace('/images/', '/eps/').split('.')[0]+'.pt', False), self.__real_img_paths))
        else:
            root_idx_dict = {osp.basename(root): idx for idx, root in enumerate(roots)}
            self.fake_paths = []
            self.real_paths = []
            for img_path in self.__fake_img_paths:
                root = osp.basename(img_path.split('/images/')[0])
                eps_root = eps_roots[root_idx_dict[root]]
                orig_root = self.roots[root_idx_dict[root]]

                dire_path = img_path.replace('/images/', '/dire/')
                dire_path = dire_path.replace(orig_root, eps_root)

                eps_path = img_path.replace('/images/', '/eps/').split('.')[0]+'.pt'
                eps_path = eps_path.replace(orig_root, eps_root)
                self.fake_paths.append((img_path, dire_path, eps_path, True))
            
            for img_path in self.__real_img_paths:
                root = osp.basename(img_path.split('/images/')[0])
                eps_root = eps_roots[root_idx_dict[root]]
                orig_root = self.roots[root_idx_dict[root]]

                dire_path = img_path.replace('/images/', '/dire/')
                dire_path = dire_path.replace(orig_root, eps_root)

                eps_path = img_path.replace('/images/', '/eps/').split('.')[0]+'.pt'
                eps_path = eps_path.replace(orig_root, eps_root)
                self.real_paths.append((img_path, dire_path, eps_path, False))


        random.shuffle(self.fake_paths)
        random.shuffle(self.real_paths)
        self.img_paths = self.fake_paths + self.real_paths
            
        img_paths = []
        for img_path, dire_path, eps_path, isfake in self.img_paths:
            if not osp.exists(eps_path) or not osp.exists(img_path) or not osp.exists(dire_path):
                continue
            try:
                Image.open(img_path)
                img_paths.append((img_path, dire_path, eps_path, isfake))
            except:
                continue
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, dire_path, eps_path, isfake = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = TF.to_tensor(img)
        # if self.istrain:
        #     # random jpeg compression
        #     img = (img * 255).to(torch.uint8)
        #     img = encode_jpeg(img, quality=random.randint(1, 100))
        #     img = decode_jpeg(img)
        #     img = img/255.
        img = img*2 - 1
        img = self.transform(img)

        dire = Image.open(dire_path).convert('RGB')
        dire = TF.to_tensor(dire)
        dire = dire*2 - 1
        dire = self.transform(dire)

        eps = torch.load(eps_path, weights_only=True, mmap=True)
        assert img.shape[1:] == dire.shape[1:] == eps.shape[1:], f"Shape mismatch: {img.shape[1:]}, {dire.shape[1:]}, {eps.shape[1:]}"
        
        
        return (img, dire, eps, isfake), (img_path, dire_path, eps_path)



class TMIMGOnlyDataset(TMDistilDireDataset):
    def __init__(self, root, istrain=True):
        super().__init__(root, prepared_dire=True)
        self.istrain=istrain

    def __getitem__(self, idx):
       
        img_path, dire_path, eps_path, isfake = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = TF.to_tensor(img)*2 - 1
        img = self.transform(img)
        eps = torch.zeros_like(img)
        dire = torch.zeros_like(img)
    
        if torch.rand(1) < 0.3 and self.istrain:
            img = TF.hflip(img)
        return (img, dire, eps, isfake), (img_path, dire_path, eps_path)
        
  


class TMEPSOnlyDataset(TMDistilDireDataset):
    def __init__(self, root, istrain=True):
        super().__init__(root, prepared_dire=True)
        img_paths = []
        for img_path, dire_path, eps_path, isfake in self.img_paths:
            if not osp.exists(eps_path) or not osp.exists(img_path):
                # print(f"File not found: {eps_path} or {img_path}")
                continue 
            try:
                eps = torch.load(eps_path, weights_only=True, mmap=True)
                img_paths.append((img_path, dire_path, eps_path, isfake))
            except Exception as e:
                print(e)
                continue
        self.img_paths = img_paths
        self.istrain=istrain

    def __getitem__(self, idx):
       
        img_path, dire_path, eps_path, isfake = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = TF.to_tensor(img)*2 - 1
        img = self.transform(img)
        eps = torch.load(eps_path, weights_only=True, mmap=True)
        dire = torch.zeros_like(img)
    
        if torch.rand(1) < 0.3 and self.istrain:
            img = TF.hflip(img)
            eps = TF.hflip(eps)
        
        return (img, dire, eps, isfake), (img_path, dire_path, eps_path)
        
    
# This is for reproducing DIRE results
class TMDireDataset(TMDistilDireDataset):
    def __init__(self, root):
        super().__init__(root, prepared_dire=True)
        
    def __getitem__(self, idx):
        img_path, dire_path, eps_path, isfake = self.img_paths[idx]
    
        dire = Image.open(dire_path).convert('RGB')
        dire = TF.to_tensor(dire)*2 - 1
        
        return (dire, isfake), (dire_path,)
        