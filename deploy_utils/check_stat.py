# create image resolution histogram

from glob import glob 
from PIL import Image
from tqdm.auto import tqdm 


tmfiles_f = glob("/truemedia-eval/images/fakes/*")
tmfiles_r = glob("/truemedia-eval/images/reals/*")

files_f = glob("/home/ubuntu/Datasets/diffusiondb/train/*")
files_r = glob("/home/ubuntu/Datasets/ffhq/in-the-wild-images/train/*")
files_f += glob("/home/ubuntu/Datasets/stylegan2-ffhq/train/*")
files_r += glob("/home/ubuntu/Datasets/coco-2017-train/train2017/train/*")
Ws_f = []
Hs_f = []
for f in tqdm(files_f):
    img = Image.open(f)
    w, h = img.size
    Ws_f.append(w)
    Hs_f.append(h)

Ws_r = []
Hs_r = []
for f in tqdm(files_r):
    img = Image.open(f)
    w, h = img.size
    Ws_r.append(w)
    Hs_r.append(h)

TMWs_r = []
TMHs_r = []
for f in tqdm(tmfiles_r):
    img = Image.open(f)
    w, h = img.size
    TMWs_r.append(w)
    TMHs_r.append(h)

TMWs_f = []
TMHs_f = []
for f in tqdm(tmfiles_f):
    img = Image.open(f)
    w, h = img.size
    TMWs_f.append(w)
    TMHs_f.append(h)

import matplotlib.pyplot as plt
import numpy as np

# plot 2d (w, h)
plt.figure(figsize=(8, 10))
plt.scatter(Ws_r, Hs_r, s=3, alpha=0.5)
plt.scatter(Ws_f, Hs_f, s=3, alpha=0.5, c='g')
plt.scatter(TMWs_r, TMHs_r, s=3, alpha=0.5, c='r')
plt.scatter(TMWs_f, TMHs_f, s=3, alpha=0.5, c='y')


plt.xlabel('Width')
plt.ylabel('Height')
plt.legend(['Training-real', 'Training-fake', 'TrueMedia-real', 'TrueMedia-fake'])
# save 
plt.savefig('resolution_hist.png')

