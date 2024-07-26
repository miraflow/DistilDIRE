from shutil import copy 
from glob import glob 
from PIL import Image
from tqdm.auto import tqdm
import os 

os.makedirs("/truemedia-eval/instagram/images/reals", exist_ok=True)
files = glob("/truemedia-eval/instagram/**/*.jpg", recursive=True)
for f in tqdm(files):
    try:
        img = Image.open(f)
    except:
        continue
    size = img.size
    if max(size) > 2000:
        continue 
    fname = f.split("/")[-1]
    copy(f, f"/truemedia-eval/instagram/images/reals/{fname}")