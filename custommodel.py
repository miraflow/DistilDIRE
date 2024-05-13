import argparse
import json
from time import perf_counter
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from utils.utils import get_distill_network
from distill_model import DistilDIRE

from PIL import Image 
import os
import torch
from utils.sanic_utils import *
import typing
import requests
import time  # Import the time module
from utils.utils import get_network, str2bool, to_cuda
from guided_diffusion.compute_dire import dire, create_argparser, dire_get_first_step_noise

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

    def __init__(self, net='DIRE', num_frames=15, data='celebahq', use_dire=True):
        self.net = net
        self.num_frames = num_frames
        rank=0
        self.use_dire=use_dire
        if not use_dire:
            self.detector =  DistilDIRE('cuda').to('cuda')
            self.detector = self._load_final_distil_state_dict(self.detector)
            self.trans = transforms.Compose(
                                        (
                                            transforms.CenterCrop(224),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
                                        )
                                    )
            args = create_argparser()
            args['timestep_respacing'] = 'ddim15'
            model, diffusion = create_model_and_diffusion(**dict_parse(args, model_and_diffusion_defaults().keys()))
        

        else:
            self.face_model = get_network("resnet50")
            self.gen_model = get_network("resnet50")
            self.face_model = self._load_state_dict(self.face_model, 'celebahq')
            self.gen_model = self._load_state_dict(self.gen_model, 'imagenet')
            
            self.trans = transforms.Compose(
                                        (
                                            # transforms.FiveCrop(128),
                                            # transforms.Lambda(lambda crops: torch.cat(crops)), # returns a 4D tensor
                                            # transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
                                        )
                                    )

        # self.gen_model = self._load_state_dict(self.gen_model, 'imagenet')
        
            args = create_argparser()
            args['timestep_respacing'] = 'ddim20'
            model, diffusion = create_model_and_diffusion(**dict_parse(args, model_and_diffusion_defaults().keys()))
        
        model.load_state_dict(torch.load(args['model_path'], map_location="cpu"))
        model.to('cuda')
        if args['use_fp16']:
            model.convert_to_fp16()
        model.eval()
        self.adm_model = model
        self.diffusion = diffusion
        self.args = args    

    def _load_final_distil_state_dict(self, net):
        state_dict = torch.load('./checkpoints/y1distil-truemedia-240408-e19.pth', map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Remove falsely saved ddp module
        net.load_state_dict({k.replace('module.', ''):state_dict[k] for k in state_dict.keys()}, strict=True)
        net.eval()
        net.to('cuda')
        print("The model is successfully loaded")
        return net

    def _load_distil_state_dict(self, net, data='celebahq'):
        state_dict = torch.load('./checkpoints/celebahq/model_epoch_4.pth', map_location="cpu") if data == 'celebahq' else torch.load('./checkpoints/imagenet/model_epoch_10.pth', map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Remove falsely saved ddp module
        net.load_state_dict({k.replace('module.', ''):state_dict[k] for k in state_dict.keys()}, strict=False)
        net.eval()
        net.to('cuda')
        print("The model is successfully loaded")
        return net


    def _load_state_dict(self, net, data='imagenet'):
        state_dict = torch.load('./celebahq_sdv2.pth', map_location="cpu") if data == 'celebahq' else torch.load('./imagenet_adm.pth', map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        net.load_state_dict(state_dict)
        net.eval()
        net.to('cuda')
        print("The model is successfully loaded")
        return net

    def _predict(self,
                 vid,
                 num_frames, save_dire=True, thr=0.4
                 ):
        img_np = extract_frames(vid, num_frames)
        # face_np, count = face_rec(img_np)
        # use_facial = True
        # try:
        #     img_tens = torch.tensor(face_np).permute(0, 3, 1, 2).float() / 255.

        # except Exception as e:
        #     print(f"Face Not detected: {str(e)}, continuing with the original image.")
        img_tens = torch.from_numpy(img_np).permute(0, 3, 1, 2).float() / 255.
        use_facial = False
        # FIX: image should be between 0 ~ 1 before normalization
        img_tens = img_tens.to('cuda')
        img_tens = torch.cat(TF.five_crop(img_tens, 256))*2 - 1
        with torch.no_grad():
            if not self.use_dire:
                eps = dire_get_first_step_noise(img_tens, self.adm_model, self.diffusion, self.args, 0)
                eps = TF.center_crop(eps, (224, 224))
                
                img_tens = self.trans(img_tens)
                prob = self.detector(img_tens, eps)['logit'].sigmoid().mean().cpu().item()
                
            else:
                dire_img, imgs, recons = dire(img_tens, self.adm_model, self.diffusion, self.args)
                dire_img = self.trans(dire_img)
                if use_facial:
                    prob = self.face_model(dire_img).sigmoid().median().cpu().item()
                else:
                    prob = self.gen_model(dire_img).sigmoid().median().cpu().item()

        # always median wins;
        return {"df_probability": prob, "prediction": real_or_fake_thres(prob, thr)}


    def _predict_wo_dire(self,
                 vid,
                 num_frames, save_dire=True, thr=0.4
                 ):
        img_np = extract_frames(vid, num_frames)
        face_np, count = face_rec(img_np)
        use_facial = True
        try:
            img_tens = torch.tensor(face_np).permute(0, 3, 1, 2).float() / 255.

        except Exception as e:
            print(f"Face Not detected: {str(e)}, continuing with the original image.")
            img_tens = torch.from_numpy(img_np).permute(0, 3, 1, 2).float() / 255.
            use_facial = False

        img_tens = transforms.CenterCrop(256)(img_tens)
        img_tens = img_tens.to('cuda')
        with torch.no_grad():
            in_tens = self.trans(img_tens)
            prob = self.model(in_tens).sigmoid()
        
        return {"df_probability": prob.median().item(), "prediction": real_or_fake_thres(prob.median().item(), thr)}


    def _forward_dire_img(self, img_path, save_dire=True, thr=0.4):
        img = Image.open(img_path).convert("RGB")
        # face_np, count = face_rec([np.array(img)])
        # use_facial = False
        # try:
        #     img_tens = torch.tensor(face_np).permute(0, 3, 1, 2).float() / 255.

        # except Exception as e:
        #     print(f"Face Not detected: {str(e)}, continuing with the original image.")
        img_tens = TF.to_tensor(img).unsqueeze(0)
        use_facial = False

        img_tens = img_tens.to('cuda')
        img_tens = torch.cat(TF.five_crop(img_tens, 256))*2 - 1
        with torch.no_grad():
            if not self.use_dire:
                eps = dire_get_first_step_noise(img_tens, self.adm_model, self.diffusion, self.args, 0)
                eps = TF.center_crop(eps, (224, 224))

                img_tens = self.trans(img_tens)
                prob = self.detector(img_tens, eps)['logit'].sigmoid().mean().cpu().item()
                
            else:
                dire_img, imgs, recons = dire(img_tens, self.adm_model, self.diffusion, self.args)
                dire_img = self.trans(dire_img)
                if use_facial:
                    prob = self.face_model(dire_img).sigmoid().median().cpu().item()
                else:
                    prob = self.gen_model(dire_img).sigmoid().median().cpu().item()
    
        return {"df_probability": prob, "prediction": real_or_fake_thres(prob, thr)}



    def _forward_dire_img_wo_dire(self, img_path, save_dire=True, thr=0.4):
        img = Image.open(img_path).convert("RGB")
        face_np, count = face_rec([np.array(img)])
        use_facial = True
        try:
            img_tens = torch.tensor(face_np).permute(0, 3, 1, 2).float() / 255.

        except Exception as e:
            print(f"Face Not detected: {str(e)}, continuing with the original image.")
            img_tens = TF.to_tensor(img).unsqueeze(0)
            use_facial = False
        
        with torch.no_grad():
            img_tens = transforms.CenterCrop(256)(img_tens)
            in_tens = self.trans(img_tens).to('cuda')
            if use_facial:
                prob = self.face_model(in_tens).sigmoid()
            else:
                prob = self.gen_model(in_tens).sigmoid()
            
        return {"df_probability": prob.median().item(), "prediction": real_or_fake_thres(prob.median().item(), thr)}


    def predict(self, inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        file_path = inputs.get('file_path', None)
        num_frames = inputs.get('num_frames', 15)
        data = inputs.get('data', 'imagenet')
        video_file = download_file(file_path)

        if os.path.isfile(video_file):
            try:
                if is_video(video_file):
                    print(f"{self.net} is being run.")
                    return self._predict(
                        video_file,
                        num_frames,
                    )
                    
                elif is_image(video_file):
                    print(f"{self.net} is being run.")
                    # return self._forward_dire_img(video_file)
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
        

    def predict_wo_dire(self, inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        file_path = inputs.get('file_path', None)
        num_frames = inputs.get('num_frames', 15)
        data = inputs.get('data', 'imagenet')
        video_file = download_file(file_path)

        if os.path.isfile(video_file):
            try:
                if is_video(video_file):
                    print(f"{self.net} is being run.")
                    return self._predict_wo_dire(
                        video_file,
                        num_frames,
                    )
                    
                elif is_image(video_file):
                    print(f"{self.net} is being run.")
                    # return self._forward_dire_img(video_file)
                    return self._forward_dire_img_wo_dire(video_file)
                
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


def main():
    """Entry point for interacting with this model via CLI."""
    start_time = perf_counter()
    curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("-p", "--file_path",
                        help="The file path for the video file to predict on", required=True, default="https://www.evalai.org/ocasio.mp4")
    parser.add_argument("-f", "--num_frames", type=int, default=15,
                        help="The number of frames to use for prediction")
    
    args = parser.parse_args()

    if args.fetch:
        CustomModel.fetch()

    # Create an instance of CustomModel using the arguments
    model = CustomModel(
        net=args.net, num_frames=args.num_frames, fp16=args.fp16)

    # Create inputs dictionary for prediction
    inputs = {
        "file_path": args.file_path,
        "fp16": args.fp16,
        "num_frames": args.num_frames,
        "net": args.net
    }
    # Call predict on the model instance with the specified arguments
    predictions = model.predict(inputs)

    # Optionally, print the predictions if you want to display them
    print(predictions)

    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()