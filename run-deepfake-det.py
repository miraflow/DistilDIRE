import argparse
import glob
import os

import torch
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from guided_diffusion.compute_dire_eps import dire
import torchvision.transforms as transforms
from torchvision.utils import save_image
from custommodel import CustomModel
from PIL import Image
from tqdm import tqdm

from utils.utils import get_network, str2bool, to_cuda

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dir", default="./testimgs", type=str, help="path to directory of images"
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="./imagenet_adm.pth",
)
parser.add_argument("--thr", type=float, default=0.2, help="threshold for binary classification")

args = parser.parse_args()

if os.path.isdir(args.dir):
    file_list = sorted(glob.glob(os.path.join(args.dir, "*.jpg")) + glob.glob(os.path.join(args.dir, "*.png"))+glob.glob(os.path.join(args.dir, "*.jpeg")))
    print(f"Testing images from '{args.dir}'")
else:
    raise FileNotFoundError(f"Invalid file path: '{args.file}'")

model = CustomModel()
print("*" * 50)
print(f"Thr: {args.thr}")


# calc TP, FP, TN, FN, Accuracy, Precision, Recall, F1 score
TP = 0
FP = 0
TN = 0
FN = 0
# Positive: deepfake, Negative: real-img
with open('result2.txt', 'w') as f:
    for img_path in tqdm(file_list, dynamic_ncols=True, disable=len(file_list) <= 1):
        p_dict = model._forward_dire_img(img_path, save_dire=True, thr=args.thr)
        f.write(f"img: {img_path} prob: {p_dict['df_probability']}\n")
        if p_dict['prediction'].lower() == 'real':
            FN += 1
            print(f"Real!: {FN}, prob: {p_dict['df_probability']}")
        else:
            TP += 1
            print(f"Fake!: {TP}, prob: {p_dict['df_probability']}")
        

    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1_score}")
   
    f.write(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}\n")
    f.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1_score}\n")

        
