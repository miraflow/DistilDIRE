from glob import glob 
import shutil
import os.path as osp 
import os 
import random 
from tqdm.auto import tqdm 


INSTA=(47_060, "/mnt/8T/instagram/images/reals")
FFHQ=(1_000, "/mnt/8T/ffhq")
MS5k_REAL=(1_940, "/mnt/8T/ms5k/reals")

MS5k_FAKE=(2_810, "/mnt/8T/ms5k/fakes")
SDFD=(0, "/mnt/8T/SDFD512")
DIFFUSION_DB=(41_190, "/mnt/8T/diffusiondb")
STLYEGAN2FFHQ=(1_000, "/mnt/8T/stylegan2-ffhq/train")
CRAWLED=(5_000, "/mnt/8T/crawled-fakes/images/fakes")

DST = "/mnt/8T/y1scale100k"
os.makedirs(DST, exist_ok=True)
os.makedirs(osp.join(DST, "images/reals"), exist_ok=True)
os.makedirs(osp.join(DST, "images/fakes"), exist_ok=True)

for i, (n, src) in tqdm(enumerate([INSTA, FFHQ, MS5k_REAL, MS5k_FAKE, SDFD, DIFFUSION_DB, STLYEGAN2FFHQ, CRAWLED])):
    files = [path for path in glob(osp.join(src, "*")) if osp.isfile(path) and path.split(".")[-1].lower() in ["jpg", "png", "jpeg", "webp"]]
    random.shuffle(files)
    files = files[:n]
    if i < 3:
        imtype="reals"
    else:
        imtype="fakes"
    for f in files:
        shutil.move(f, osp.join(DST, "images", imtype, osp.basename(f)))
    print(f"Moved {len(files)} images from {src} to {DST}/images/{imtype}")

