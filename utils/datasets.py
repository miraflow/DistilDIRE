import os
from io import BytesIO

from torch.utils.data import Dataset
import random
import os.path as osp
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
from copy import deepcopy
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data.sampler import WeightedRandomSampler

from torch.utils.data.distributed import DistributedSampler

from guided_diffusion.compute_dire import dire, create_argparser, dire_get_first_step_noise

from utils.config import CONFIGCLASS

ImageFile.LOAD_TRUNCATED_IMAGES = True


def dataset_folder(root: str, phase, adm_model, diffusion, cfg: CONFIGCLASS, adm_args):
    if cfg.mode == "binary":
        return binary_distill_dataset(root, phase, adm_model, diffusion, cfg, adm_args)
    if cfg.mode == "filename":
        return FileNameDataset(root, cfg)
    raise ValueError("cfg.mode needs to be binary or filename.")

def custom_random_crop(img, left, top, size):
    if isinstance(img, Image.Image):
        # PIL 이미지인 경우
        right = left + size
        bottom = top + size
        cropped_img = img.crop((left, top, right, bottom))
    elif isinstance(img, torch.Tensor):
        # PyTorch 텐서인 경우
        b, channels, height, width = img.size()
        left = max(0, left)
        top = max(0, top)
        right = min(width, left + size)
        bottom = min(height, top + size)
        cropped_img = img[:, top:bottom, left:right]
    else:
        raise TypeError("Unsupported input type. Only PIL Image or PyTorch Tensor is supported.")

    return cropped_img

def custom_to_tensor(img):
    # check if the img is tensor
    if isinstance(img, torch.Tensor):
        return img
    # PIL image to tensor
    return TF.to_tensor(img)

def custom_normalize(img):
    return TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_transforms(cfg):
    identity_transform = transforms.Lambda(lambda img: img)

    if cfg.isTrain or cfg.aug_resize:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, cfg))
    else:
        rz_func = identity_transform
        
    width, height = 224, 224

    if cfg.isTrain:
        crop_func = transforms.RandomCrop(cfg.cropSize)
        # crop_func = transforms.Lambda(lambda img: custom_random_crop(img, random_left, random_top, cfg.cropSize))
    else:
        crop_func = transforms.CenterCrop(cfg.cropSize) if cfg.aug_crop else identity_transform

    randomness = random.uniform(0, 1)
    if cfg.isTrain and cfg.aug_flip and randomness >= 0.5:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = identity_transform
        
    transform= transforms.Compose(
                [
                    rz_func,
                    transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
                    crop_func,
                    flip_func,
                    transforms.Lambda(lambda img: custom_to_tensor(img)),
                ]
            )
    
    return transform

class BinaryClassificationDataset(Dataset):
    def __init__(self, root, phase, cfg):
        self.root = root
        self.phase = phase
        self.cfg = cfg
        
        self.data = self.load_data('celebahq') + self.load_data('imagenet')
        
    def load_data(self, dataset_name):
        
        labels = {"reals": 0, "fakes": 1}
        # labels = {"fakes":1}
        root = osp.join(self.root, dataset_name)
        if dataset_name=='celebahq':
            gen_model_names = ['dalle2', 'if', 'midjourney', 'sdv2']
        else:
            gen_model_names = ['adm', 'sdv1']
        data = []   
        for gen_model_name in gen_model_names:
            for label, label_idx in labels.items():
                if self.phase == 'test' and label == 'fakes':
                    folder_path = os.path.join(root, 'images', self.phase, label, gen_model_name)
                    dire_folder_path = os.path.join(root, "dire", self.phase, label, gen_model_name)
                    eps_folder_path = os.path.join(root, "eps", self.phase, label, gen_model_name)
                    
                elif self.phase == 'train' and label == 'fakes' and 'imagenet' in root:
                    folder_path = os.path.join(root, 'images', self.phase, label, 'adm')
                    dire_folder_path = os.path.join(root, "dire", self.phase, label, 'adm')  
                    eps_folder_path = os.path.join(root, "eps", self.phase, label, 'adm')
                            
                else:
                    folder_path = os.path.join(root, 'images', self.phase, label)
                    dire_folder_path = os.path.join(root, "dire", self.phase, label)
                    eps_folder_path = os.path.join(root, "eps", self.phase, label)
                
                # if "imagenet" in root:
                #     eps_folder_path = eps_folder_path.replace("datasets", "datasets_ext")
                
                if (self.phase == 'train' and 'imagenet' in root) or (self.phase == 'test' and 'imagenet' in root): # self.phase == 'train' and 
                    # 'adm' 폴더 안에 있는 서브폴더 리스트 가져오기
                    subfolder_list = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                    
                    # 각 서브폴더에 있는 이미지들을 가져와서 데이터에 추가
                    for subfolder in subfolder_list:
                        subfolder_path = os.path.join(folder_path, subfolder)
                        dire_subfolder_path = os.path.join(dire_folder_path, subfolder)
                        eps_subfolder_path = os.path.join(eps_folder_path, subfolder)
                        
                        image_files = sorted(os.listdir(subfolder_path))
                        if not osp.exists(dire_subfolder_path) or not osp.exists(eps_subfolder_path):
                            continue
                        eps_files = sorted(os.listdir(eps_subfolder_path))
                        dire_files = sorted(os.listdir(dire_subfolder_path))
                        
                        for image_file, dire_file, eps_file in zip(image_files, dire_files, eps_files):
                            ext = image_file.split('.')[-1].lower()
                            if ext=='jpg' or ext=='png' or ext=='jpeg':
                                image_path = os.path.join(subfolder_path, image_file)
                                dire_image_path = os.path.join(dire_subfolder_path, dire_file)
                                eps_file_path = os.path.join(eps_subfolder_path, eps_file)
                                
                                if not (os.path.exists(image_path) and os.path.exists(dire_image_path) and os.path.exists(eps_file_path)):
                                    continue
                                data.append((image_path, dire_image_path, eps_file_path, label_idx))
                        
                
                else:
                    image_files = sorted(os.listdir(folder_path))
                    dire_files = sorted(os.listdir(dire_folder_path))
                    eps_files = sorted(os.listdir(eps_folder_path))
                    
                    for image_file, dire_file, eps_file in zip(image_files, dire_files, eps_files):
                        if image_file.endswith(".jpg") or image_file.endswith(".png"):
                            image_path = os.path.join(folder_path, image_file)
                            dire_image_path = os.path.join(dire_folder_path, dire_file)
                            eps_file_path = os.path.join(eps_folder_path, eps_file)
                            if not (os.path.exists(image_path) and os.path.exists(dire_image_path) and os.path.exists(eps_file_path)):
                                continue
                            data.append((image_path, dire_image_path, eps_file_path, label_idx))

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' img for adm infernce: should be normalized between -1 and 1 (not using img net mean and std)
            img for dire inference: should be normalized between -1 and 1 (using img net mean and std)
            # ========================================================================================
            dire for dire inference: should be normalized between -1 and 1 (using img net mean and std)
        '''
        image_path, dire_image_path, eps_file_path, label = self.data[idx]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        dire_image = cv2.cvtColor(cv2.imread(dire_image_path), cv2.COLOR_BGR2RGB)
        # TODO: we have to fix it before releasing
        try:
            eps = torch.load(eps_file_path, map_location='cpu')
        except:
            return self.__getitem__((idx+1)%len(self.data))
        
        image = torchvision.transforms.ToTensor()(image)
        dire_image = torchvision.transforms.ToTensor()(dire_image)
        base_transform = transforms.Resize((224, 224))
        norm_transform = transforms.Lambda(lambda img: custom_normalize(img=img))
        
        # cat = torch.cat([transforms.Resize((224, 224))(image.unsqueeze(0)), 
        #                  transforms.Resize((224, 224))(dire_image.unsqueeze(0))], dim=0)
        # cat = base_transform(cat)
        # un_norm_img, dire_image = cat[0], cat[1]
        image = base_transform(image)
        dire_image = base_transform(dire_image)
        eps = base_transform(eps)
        
        image = norm_transform(image)
        dire_image = norm_transform(dire_image)
        
        if torch.rand(1) < 0.5 and self.phase == 'train':
            image = TF.hflip(image)
            dire_image = TF.hflip(dire_image)
            eps = TF.hflip(eps)
        
        return image, dire_image, eps, label
        
        

def binary_distill_dataset(root: str, phase, cfg: CONFIGCLASS):
    
    # TODO
    # Choose dataset based on the given root directory
    data_path = root
    dataset = BinaryClassificationDataset(data_path, phase, cfg)
    
    return dataset
    
    
def get_binary_distill_dataloader(cfg: CONFIGCLASS, phase, distributed):
    dataset = binary_distill_dataset(cfg.dataset_root, phase, cfg)
    
    # Use DistributedSampler for distributed training
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
    
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=False,  # Shuffle is done by DistributedSampler
            sampler=sampler,  # Use DistributedSampler
            num_workers=cfg.num_workers
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )
    
    return dataloader


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def blur_jpg_augment(img: Image.Image, cfg: CONFIGCLASS):
    # if img type is tensor
    if isinstance(img, torch.Tensor):
        return img
    img: np.ndarray = np.array(img)
    if cfg.isTrain:
        if random.random() < cfg.blur_prob:
            sig = sample_continuous(cfg.blur_sig)
            gaussian_blur(img, sig)

        if random.random() < cfg.jpg_prob:
            method = sample_discrete(cfg.jpg_method)
            qual = sample_discrete(cfg.jpg_qual)
            img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list):
    return s[0] if len(s) == 1 else random.choice(s)


def gaussian_blur(img: np.ndarray, sigma: float):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    interp = sample_discrete(cfg.rz_interp)
    return TF.resize(img, cfg.loadSize, interpolation=rz_dict[interp])


def get_dataset(cfg: CONFIGCLASS):
    dset_lst = []
    for dataset in cfg.datasets:
        root = os.path.join(cfg.dataset_root, dataset)
        dset = dataset_folder(root, cfg)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset: torch.utils.data.ConcatDataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(cfg: CONFIGCLASS):
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False
    dataset = get_dataset(cfg)
    sampler = get_bal_sampler(dataset) if cfg.class_bal else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
    )
