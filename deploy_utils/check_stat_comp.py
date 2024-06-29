# create image resolution histogram

from glob import glob 
from PIL import Image
from tqdm.auto import tqdm 
import os

tmfiles_f = glob("/home/ubuntu/y1/DistilDIRE/datasets/truemedia-political/images/fakes/*")
tmfiles_r = glob("/home/ubuntu/y1/DistilDIRE/datasets/truemedia-political/images/reals/*")

files_f = glob("/home/ubuntu/y1/DistilDIRE/datasets/y1-global-truemedia/images/fakes/*")
files_r = glob("/truemedia-eval/y1dataset/images/reals/*")
# files_f += glob("/home/ubuntu/Datasets/stylegan2-ffhq/train/*")
# files_r += glob("/home/ubuntu/Datasets/coco-2017-train/train2017/train/*")
training_f = []
fake_cnt = 0
real_cnt = 0
for f in tqdm(files_f):
    b = os.stat(f).st_size
    img = Image.open(f)
    w, h = img.size
    if (b/(w*h)) < 1:
        fake_cnt += 1
    training_f.append(b/(w*h))

training_r = []
for f in tqdm(files_r):
    try:
        img = Image.open(f)
        b = os.stat(f).st_size
        w, h = img.size
        if (b/(w*h)) < 1:
            real_cnt += 1
        training_r.append(b/(w*h))
    except:
        continue

TM_r = []
for f in tqdm(tmfiles_r):
    try:
        img = Image.open(f)
        b = os.stat(f).st_size
        w, h = img.size
        TM_r.append(b/(w*h))
    except:
        continue

TM_f = []
for f in tqdm(tmfiles_f):
    try:
        img = Image.open(f)
        b = os.stat(f).st_size
        w, h = img.size
        TM_f.append(b/(w*h))
    except:
        continue

import matplotlib.pyplot as plt
import numpy as np

# plot 2d (w, h)
plt.figure(figsize=(8, 6))
plt.hist(training_r, bins=100, alpha=0.5)
plt.hist(training_f, bins=100, alpha=0.5, color='g')
plt.hist(TM_r, bins=100, alpha=0.5, color='r')
plt.hist(TM_f, bins=100, alpha=0.5, color='y')

# Bytes/pixel 
plt.xlabel('Bytes/pixel')
plt.legend(['Training-real', 'Training-fake', 'TrueMedia-real', 'TrueMedia-fake'])
# save 
plt.savefig('comp_hist.png')
print(f"Training real: {real_cnt}, Training fake: {fake_cnt}")

