"""
Modified from guided-diffusion/scripts/image_sample.py
"""

import argparse
import os
import torch

import sys
import cv2

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th


from .guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=-1,
        use_ddim=True,
        model_path="./256x256-adm.pt",
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    sanic_dict = dict(
       attention_resolutions='32,16,8',
       class_cond=False,
       diffusion_steps=1000,
       dropout=0.1,
       image_size=256,
       learn_sigma=True,
       noise_schedule='linear',
       num_channels=256,
       num_head_channels=64,
       num_res_blocks=2,
       resblock_updown=True,
       use_fp16=False,
       use_scale_shift_norm=True,
    )
    defaults.update(sanic_dict)
    
    return defaults


@torch.no_grad()
def dire(img_batch:torch.Tensor, model, diffusion, args):
    

    print("computing recons & DIRE ...")

    imgs = img_batch.cuda()
    
    batch_size = imgs.shape[0]
    model_kwargs = {}

    reverse_fn = diffusion.ddim_reverse_sample_loop
    
    # imgs = reshape_image(imgs, args['image_size'])
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
    # dire = dire.clamp(0, 0.9).to(th.float32)
    
    # scale imgs and recons
    imgs = (imgs+1)*0.5
    recons = (recons+1)*0.5

    return dire, imgs, recons

@torch.no_grad()
def dire_get_first_step_noise(img_batch:torch.Tensor, model, diffusion, args, device):
    # print("computing first step noise[] ...")

    imgs = img_batch.to(device)
    # print(f"Device: {imgs.get_device()}")
    
    batch_size = imgs.shape[0]
    model_kwargs = {}

    reverse_fn = diffusion.ddim_reverse_sample_only_eps
    imgs = reshape_image(imgs, args['image_size'])
    
    eps = reverse_fn(
        model,
        # (batch_size, 3, args['image_size'], args['image_size']),
        x=imgs,
        t=torch.zeros(imgs.shape[0],).int().to(device),
        clip_denoised=args['clip_denoised'],
        model_kwargs=model_kwargs,
        eta=0.0
        #real_step=args['real_step'],
    )

    return eps

