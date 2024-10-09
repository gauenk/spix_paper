# -- tensor opts --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- drawing help --
import torchvision.utils as tv_utils
from skimage.segmentation import mark_boundaries
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes,draw_segmentation_masks

# -- basics --
from pathlib import Path
from easydict import EasyDict as edict

# -- dev --
from dev_basics.utils.metrics import compute_psnrs,compute_ssims

# -- superpixel --
from spix_paper import data
from spix_paper import utils
from spix_paper.attn import SuperpixelAttention
from spix_paper.attn import NeighAttnMat,NeighAttnAgg
from spix_paper.connected import connected_sp
from spix_paper.spix_utils import run_slic
from spix_paper.attn import run_search_fxn,run_stnls_search
from spix_paper.attn import run_stnls_agg,run_nat_agg
from spix_paper import isp

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def interp(img,tH,tW):
    img = TF.resize(img,(tH,tW),interpolation=InterpolationMode.NEAREST)
    return img

def divBy2(img):
    H,W = img.shape[-2:]
    return interp(img,H//2,W//2)

def save_image_r(img,fn):
    img = rearrange(img,'... h w c -> ... c h w')
    save_image(img,fn)

def main():

    root = Path("output/prepare_isp/")
    if not root.exists(): root.mkdir(parents=True)
    # vid_names = ["tennis"]
    vid_names = ["tuk-tuk"]
    img0 = data.davis_example(isize=None,nframes=10,vid_names=vid_names)
    img0 = divBy2(img0[0,:1,:,:480,:480]).to("cuda")
    print("img0.shape: ",img0.shape)
    save_image(img0,root/"img0.png")
    img0 = rearrange(img0,'b c h w -> b h w c').cpu()
    img0 = repeat(img0,'1 h w c -> r h w c',r=5)
    raw,noise_info = isp.run_unprocess(img0)
    keys = ['red_gain','blue_gain','cam2rgb']
    args = [noise_info[k] for k in keys]
    print(raw.shape,raw.max(),raw.min())
    # save_image(raw,root/"raw.png")
    raw = raw.mean(0,keepdim=True).repeat(5,1,1,1)
    img1 = isp.run_process(raw,*args)
    save_image_r(img1,root/"img1.png")
    delta = th.mean((img0 - img1)**2).item()
    print("difference: ",delta)


if __name__ == "__main__":
    main()
