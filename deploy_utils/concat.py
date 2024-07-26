import shutil
from tqdm.auto import tqdm
import os.path as osp
from glob import glob
from PIL import Image

# PATH = "/home/ubuntu/Datasets/"
ROOTS = ['/home/ubuntu/Datasets/coco-2017-train/train2017/train', "/home/ubuntu/Datasets/ms-5kimages", "/home/ubuntu/Datasets/ffhq/in-the-wild-images/train"]
names = ['coco', 'ms-5k', 'ffhq']
DEST = "/home/ubuntu/y1/y1-global-truemedia/train/images"
skipped = 0
for name, root in zip(names, ROOTS):
    # cur_path = osp.join(root, 'train/images')
    # fakes = glob(osp.join(cur_path, 'fakes', '*'))
    reals = glob(osp.join(root, '*'))
    # for fake in fakes:
    #     fname = f'{root}_{osp.basename(fake)}'
    #     if not osp.exists(osp.join(DEST, 'fakes', fname)):
    #         shutil.copy(fake, osp.join(DEST, 'fakes', fname))
    if name == 'coco':
        reals = reals[:35000]
    for real in tqdm(reals):
        fname = f'{name}_{osp.basename(real)}'
        size = Image.open(real).size
        if max(size) > 2000:
            skipped += 1
            continue
        if not osp.exists(osp.join(DEST, 'reals', fname)):
            shutil.copy(real, osp.join(DEST, 'reals', fname))

print(f"Skipped {skipped} images for being too large.")
        