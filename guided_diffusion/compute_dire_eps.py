"""
Modified from guided-diffusion/scripts/image_sample.py
"""

import argparse
import os
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset import TMDistilDireDataset
from torch.utils.data import DataLoader
import cv2

import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm.auto import tqdm 

import numpy as np
import torch as th
import torchvision
import os.path as osp


from .guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    dict_parse,
    args_to_dict,

)



def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic", antialias=True)
    return imgs



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=-1,
        use_ddim=True,
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    sanic_dict = dict(
       attention_resolutions='32,16,8',
       class_cond=False,
       diffusion_steps=1000,
       image_size=256,
       learn_sigma=True,
       model_path="./models/256x256-adm.pt",
       noise_schedule='linear',
       num_channels=256,
       num_head_channels=64,
       num_res_blocks=2,
       resblock_updown=True,
       use_fp16=True,
       use_scale_shift_norm=True,
       data_root="",
       compute_dire=False,
       compute_eps=False,
       save_root="",
       batch_size=32,
    )
    defaults.update(sanic_dict)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    return args_to_dict(args, list(defaults.keys()))

def create_dicts_for_static_init():
    defaults = dict(
        clip_denoised=True,
        num_samples=-1,
        use_ddim=True,
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    sanic_dict = dict(
       attention_resolutions='32,16,8',
       class_cond=False,
       diffusion_steps=1000,
       image_size=256,
       learn_sigma=True,
       model_path="./models/256x256-adm.pt",
       noise_schedule='linear',
       num_channels=256,
       num_head_channels=64,
       num_res_blocks=2,
       resblock_updown=True,
       use_fp16=True,
       use_scale_shift_norm=True,
       data_root="",
       compute_dire=False,
       compute_eps=False,
       save_root="",
       batch_size=32,
    )
    defaults.update(sanic_dict)

    return defaults



@torch.no_grad()
def dire(img_batch:torch.Tensor, model, diffusion, args, save_img=False, save_path=None):
    print("computing recons & DIRE ...")
    imgs = img_batch.cuda()
    
    batch_size = imgs.shape[0]
    model_kwargs = {}

    reverse_fn = diffusion.ddim_reverse_sample_loop
    assert (imgs.shape[2] == imgs.shape[3]) and (imgs.shape[3] == args['image_size']), f"Image size mismatch: {imgs.shape[2]} != {args['image_size']}"
    latent = reverse_fn(
        model,
        (batch_size, 3, args['image_size'], args['image_size']),
        noise=imgs,
        clip_denoised=args['clip_denoised'],
        model_kwargs=model_kwargs,
        real_step=args['real_step'],
    )
    sample_fn = diffusion.p_sample_loop if not args['use_ddim'] else diffusion.ddim_sample_loop
    recons = sample_fn(
        model,
        (batch_size, 3, args['image_size'], args['image_size']),
        noise=latent,
        clip_denoised=args['clip_denoised'],
        model_kwargs=model_kwargs,
        real_step=args['real_step']
    )

    dire = th.abs(imgs - recons)
    dire = (dire*255./2).clamp(0, 255).to(th.uint8)
    dire = dire.contiguous() / 255.
    dire = (dire).clamp(0, 1).to(th.float32)
    
    # scale imgs and recons
    imgs = (imgs+1)*0.5
    recons = (recons+1)*0.5

    if save_img:
        # save images
        for i in range(len(img_batch)):
            dire_img = dire[i].detach().cpu().numpy().transpose(1, 2, 0)
            dire_img = cv2.cvtColor(dire_img, cv2.COLOR_RGB2BGR)
            
            dire_path = os.path.join(save_path, f"dire_{i}.png")
            cv2.imwrite(dire_path, dire_img*255)

    return dire, imgs, recons

@torch.no_grad()
def dire_get_first_step_noise(img_batch:torch.Tensor, model, diffusion, args, device):
    imgs = img_batch.to(device)    
    batch_size = imgs.shape[0]
    model_kwargs = {}

    reverse_fn = diffusion.ddim_reverse_sample_only_eps
    assert (imgs.shape[2] == imgs.shape[3]) and (imgs.shape[3] == args['image_size']), f"Image size mismatch: {imgs.shape[2]} != {args['image_size']}"
    
    eps = reverse_fn(
        model,
        # (batch_size, 3, args['image_size'], args['image_size']),
        x=imgs,
        t=torch.zeros(imgs.shape[0],).long().to(device),
        clip_denoised=args['clip_denoised'],
        model_kwargs=model_kwargs,
        eta=0.0
        #real_step=args['real_step'],
    )

    return eps


if __name__ == "__main__":
    from torch.utils.data.distributed import DistributedSampler
    from s3torchconnector import S3MapDataset, S3IterableDataset, S3Checkpoint

    import torch.distributed as dist
    import os 
    
    dist.init_process_group(backend='nccl', init_method='env://')
    
    local_rank = int(os.environ['LOCAL_RANK']) 
    torch.cuda.set_device(local_rank)
    DATASET_URI="s3://truemedia-dataset/distil-dire-dataset/y1scale100k"
    REGION = "us-west-2"
    checkpoint = S3Checkpoint(region=REGION)
    # Set device for this process
    device = torch.device("cuda") 

    adm_args = create_argparser()
    adm_args['timestep_respacing'] = 'ddim20'
    adm_model, diffusion = create_model_and_diffusion(**dict_parse(adm_args, model_and_diffusion_defaults().keys()))
    print(f"checkpoint: {adm_args['model_path']}")
    print(f"model channel: {adm_args['num_channels']}, {adm_model.model_channels}")
    adm_model.load_state_dict(torch.load(adm_args['model_path'], map_location="cpu"))
    adm_model.to(device)

    adm_model.convert_to_fp16()
    adm_model.eval()

    dataset = TMDistilDireDataset(adm_args['data_root'], prepared_dire=False)
    sampler = DistributedSampler(dataset, shuffle=False)
    os.makedirs(osp.join(adm_args['save_root'], 'images', 'fakes'), exist_ok=True)
    os.makedirs(osp.join(adm_args['save_root'], 'images', 'reals'), exist_ok=True)
    os.makedirs(osp.join(adm_args['save_root'], 'dire', 'fakes'), exist_ok=True)
    os.makedirs(osp.join(adm_args['save_root'], 'dire', 'reals'), exist_ok=True)
    os.makedirs(osp.join(adm_args['save_root'], 'eps', 'fakes'), exist_ok=True)
    os.makedirs(osp.join(adm_args['save_root'], 'eps', 'reals'), exist_ok=True)
    print(f"Dataset length: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=adm_args['batch_size'], num_workers=4, drop_last=False, pin_memory=True, sampler=sampler)#

    for (img_batch, dire_batch, eps_batch, isfake_batch), (img_pathes, dire_pathes, eps_pathes) in tqdm(dataloader):
        haveall=True 
        for i in range(len(img_batch)):
            basename = osp.basename(img_pathes[i])
            isfake = isfake_batch[i]
            
            img_path = osp.join(adm_args['save_root'], 'images', 'fakes', basename) if isfake else osp.join(adm_args['save_root'], 'images', 'reals', basename)
            dire_path = img_path.replace('/images/', '/dire/')
            eps_path = img_path.replace('/images/', '/eps/').split('.')[0] + '.pt'
            if (not osp.exists(img_pathes[i])) or (not osp.exists(dire_path)) or (not osp.exists(eps_path)):
                haveall=False
                break
        if haveall:
            continue
        with torch.no_grad():
            eps = None
            img = (img_batch.detach().cpu()+1)*0.5
            if adm_args['compute_eps']:
                eps = dire_get_first_step_noise(img_batch, adm_model, diffusion, adm_args, device)
                eps = eps.detach().cpu()

            if adm_args['compute_dire']:
                dire_img, img, recons = dire(img_batch, adm_model, diffusion, adm_args)
                dire_img = dire_img.detach().cpu()
                img = img.detach().cpu()
            
            for i in range(len(img_batch)):
                basename = osp.basename(img_pathes[i])
                isfake = isfake_batch[i]
                
                img_path = osp.join(adm_args['save_root'], 'images', 'fakes', basename) if isfake else osp.join(adm_args['save_root'], 'images', 'reals', basename)
                dire_path = img_path.replace('/images/', '/dire/')
                eps_path = img_path.replace('/images/', '/eps/').split('.')[0] + '.pt'

                dire_path = osp.join(DATASET_URI, 'dire', dire_path.split('/dire/')[-1])
                eps_path = osp.join(DATASET_URI, 'eps', eps_path.split('/eps/')[-1])
                
                with checkpoint.writer(dire_path) as writer:
                    img = transforms.ToPILImage()(dire_img[i])
                    img.save(writer, format='png')
                    # torchvision.utils.save_image(dire_img[i], reader, format=dire_path.split('.')[-1])
                # if not osp.exists(dire_path) and adm_args['compute_dire']:
                #     torchvision.utils.save_image(dire_img[i], dire_path)

                with checkpoint.writer(eps_path) as writer:
                    torch.save(eps[i], writer)

                # if not osp.exists(img_path):
                #     torchvision.utils.save_image(img[i], img_path)
                
                # if not osp.exists(eps_path) and eps is not None:
                #     torch.save(eps[i], eps_path)