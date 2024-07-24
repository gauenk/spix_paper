# -- basics --
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict
from torchvision.utils import save_image,make_grid
from pathlib import Path

# -- data --
import data_hub

def davis_example(nframes=5,isize=480,vid_names=None):

    # -- data config --
    dcfg = edict()
    dcfg.dname = "davis"
    dcfg.dset = "train"
    dcfg.sigma = 1.
    dcfg.nframes = nframes
    dcfg.isize = isize

    # -- load images --
    device = "cuda:0"
    data, loaders = data_hub.sets.load(dcfg)
    if vid_names is None:
        # vid_names = ["elephant",]
        # vid_names = ["bmx-bumps","elephant","boxing-fisheye","dancing"]
        # vid_names = ["longboard","boxing-fisheye","dancing"]
        # vid_names = ["surf","boxing-fisheye","dancing"]
        # vid_names = ["bus","boxing-fisheye","dancing"]
        # vid_names = ["dancing"]
        vid_names = ["drone","boxing-fisheye","dancing"]
    # isel = {"bmx-bumps":[150,240],"boxing-fisheye":[200,200]}
    vid = []
    for name in vid_names:
        _index = data_hub.filter_subseq(data.tr,name,frame_start=0,
                                        frame_end=nframes-1)[0]
        _vid = data.tr[_index]['clean']/255.
        # if name in isel: sh,sw = isel[name]
        # else: sh,sw = 0,0
        # _vid = _vid[:,:,sh:sh+240,sw:sw+240].to(device)
        vid.append(_vid.to(device))
    vid = th.stack(vid)

    return vid

