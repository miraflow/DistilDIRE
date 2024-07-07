import argparse
import json
from time import perf_counter
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.io import decode_jpeg, encode_jpeg


from PIL import Image 
import os
import torch
from utils.sanic_utils import *
import typing
import requests
import time  # Import the time module
from guided_diffusion.compute_dire_eps import dire_get_first_step_noise, create_argparser
from networks.distill_model import DistilDIREOnlyEPS
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    dict_parse
)


def download_file(input_path):
    """
    Download a file from a given URL and save it locally if input_path is a URL.
    If input_path is a local file path and the file exists, skip the download.

    :param input_path: The URL of the file to download or a local file path.
    :return: The local filepath to the downloaded or existing file.
    """
    # Check if input_path is a URL
    if input_path.startswith(('http://', 'https://')):
        # Extract filename from the URL
        # Splits the URL by '/' and get the last part
        filename = input_path.split('/')[-1]

        # Ensure the filename does not contain query parameters if present in the URL
        # Splits the filename by '?' and get the first part
        filename = filename.split('?')[0]

        # put jpg extension if not present
        if '.' not in filename:
            filename += ".jpg"

        # Define the local path where the file will be saved
        local_filepath = os.path.join('.', filename)

        # Check if file already exists locally
        if os.path.isfile(local_filepath):
            print(f"The file already exists locally: {local_filepath}")
            return local_filepath

        # Start timing the download
        start_time = time.time()

        # Send a GET request to the URL
        response = requests.get(input_path, stream=True)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Open the local file in write-binary mode
        with open(local_filepath, 'wb') as file:
            # Write the content of the response to the local file
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # End timing the download
        end_time = time.time()

        # Calculate the download duration
        download_duration = end_time - start_time

        print(
            f"Downloaded file saved to {local_filepath} in {download_duration:.2f} seconds.")

    else:
        # Assume input_path is a local file path
        local_filepath = input_path
        # Check if the specified local file exists
        if not os.path.isfile(local_filepath):
            raise FileNotFoundError(f"No such file: '{local_filepath}'")
        print(f"Using existing file: {local_filepath}")

    return local_filepath


class CustomModel:
    """Wrapper for a DIRE model."""

    def __init__(self, net='DIRE', num_frames=15, ckpt='truemedia-global-scaled.pth'):
        self.net = net
        self.num_frames = num_frames
        
        self.model =  DistilDIREOnlyEPS('cuda').to('cuda')
        self.trans = transforms.Compose((transforms.Resize(256), transforms.CenterCrop((256, 256)),))
        
        self._load_state_dict(ckpt)
        
        args = create_argparser()
        args['timestep_respacing'] = 'ddim20'
        adm_model, diffusion = create_model_and_diffusion(**dict_parse(args, model_and_diffusion_defaults().keys()))
        adm_model.load_state_dict(torch.load(args['model_path'], map_location="cpu"))
        adm_model.cuda()
        adm_model.convert_to_fp16()
        adm_model.eval()
        self.adm_model = adm_model
        self.diffusion = diffusion
        self.args = args    


    def _load_state_dict(self, ckpt):
        print(f"Loading the model from {ckpt}...")
        state_dict = torch.load(ckpt, map_location="cpu")['model'] 
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.cuda()
        print("The model is successfully loaded")


    def _forward_dire_img(self, img_path, save_dire=True, thr=0.5):
        img = Image.open(img_path).convert("RGB")
        # w, h = img.size
        # fsize = os.stat(img_path).st_size
        # img = (TF.to_tensor(img)*255).to(torch.uint8)

        # comp = fsize/(w*h)
        # comp_quality = min(0.1/comp * 100, 100)
        # comp_quality = max(comp_quality, 1)
        # img = decode_jpeg(encode_jpeg(img, quality=int(comp_quality)))
        # IMG = TF.to_pil_image(img)
        # IMG.save("compressed.jpg")
        # img = img / 255.
        # print(comp_quality)
        img = TF.to_tensor(img)
        img = self.trans(img).cuda() * 2 - 1
        print(f"Min: {img.min()}, Max: {img.max()}")
        img = img.unsqueeze(0)
        with torch.no_grad():
            eps = dire_get_first_step_noise(img, self.adm_model, self.diffusion, self.args, "cuda")
            prob = self.model(eps)['logit'].sigmoid()
            
        return {"df_probability": prob.median().item(), "prediction": real_or_fake_thres(prob.median().item(), thr)}


    def predict(self, inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        file_path = inputs.get('file_path', None)
        video_file = download_file(file_path)

        if os.path.isfile(video_file):
            try:
                if is_image(video_file):
                    print(f"{self.net} is being run.")
                    return self._forward_dire_img(video_file)
                
                else:
                    print(
                        f"Invalid media file: {video_file}. Please provide a valid video/img file.")
                    return {"msg":  f"Invalid media file: {video_file}. Please provide a valid video/img file."}
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return {"msg": f"An error occurred: {str(e)}"}
        else:
            print(f"The file {video_file} does not exist.")
            return {"msg": f"The file {video_file} does not exist."}
        

    @classmethod
    def fetch(cls) -> None:
        cls()

