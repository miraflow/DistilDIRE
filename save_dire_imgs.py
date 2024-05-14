import torch 
from torchvision.utils import save_image
from guided_diffusion.compute_dire_eps import dire
from glob import glob 
from tqdm.auto import tqdm
from PIL import Image
from torchvision.transforms import functional as TF 
import numpy as np
import os.path as osp


from guided_diffusion.compute_dire_eps import dire, create_argparser,dire_get_first_step_noise
from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, dict_parse

dire_root = '/workspace/truemedia-dire/reals/DIRE'
img_root = '/workspace/truemedia-dire/reals/images'

device = 'cuda'
adm_args = create_argparser()
adm_args['timestep_respacing'] = 'ddim15'
adm_model, diffusion = create_model_and_diffusion(**dict_parse(adm_args, model_and_diffusion_defaults().keys()))
adm_model.load_state_dict(torch.load(adm_args['model_path'], map_location="cpu"))
adm_model.to(device)

print(f"Loaded ADM model from {adm_args['model_path']}")
imgs = glob("/workspace/reals/*")
for im_path in tqdm(imgs, desc='Calculating DIRE images', total=len(imgs)):
    exist=True
    for j in range(10):
        if not osp.exists(osp.join(img_root, f"{osp.basename(im_path).split('.')[0]}_imgs_{j}.png")):
            exist=False
            break
    if exist:
        continue
    try:
        img_np = np.array(Image.open(im_path).convert('RGB'))[np.newaxis, ...]
    
        img_tens = torch.from_numpy(img_np).permute(0, 3, 1, 2).float() / 255.
        img_tens = img_tens*2 - 1 
        # img_tens = TF.resize(img_tens, (256, 256))
        # img_tens = TF.center_crop(img_tens, (256, 256))
        img_tens = torch.cat(TF.ten_crop(img_tens, (256, 256)))
    except:
        continue
    img_tens = img_tens.cuda().float()
    with torch.no_grad():
        dire_im, imgs, recons = dire(img_tens, adm_model, diffusion, adm_args)
        imgs = imgs.cpu()
        recons = recons.cpu()
        dire_im = dire_im.cpu()
        for j in range(len(img_tens)):
            save_image(imgs[j], osp.join(img_root, f"{osp.basename(im_path).split('.')[0]}_imgs_{j}.png"))
            save_image(dire_im[j], osp.join(dire_root, f"{osp.basename(im_path).split('.')[0]}_dire_{j}.png"))
