

import numpy as np
from scipy.io import loadmat
import torch as th

from dev_basics.utils.metrics import compute_psnrs,compute_ssims


def compute_asa(sp,gt):

    # -- prepare --
    if not th.is_tensor(sp):
        sp = th.from_numpy(sp*1.).long()
    if not th.is_tensor(gt):
        gt = th.from_numpy(gt*1.).long()

    # -- normalize --
    gt = (gt*1.).long()-int(gt.min().item())

    # -- unpack --
    device = sp.device
    H,W = sp.shape

    # -- allocate hist --
    Nsp = max(len(sp.unique()),int(sp.max()+1))
    Ngt = max(len(gt.unique()),int(gt.max()+1))
    hist = th.zeros(Nsp*Ngt,device=device)

    # -- fill hist --
    inds = sp.ravel()*Ngt+gt.ravel()
    ones = th.ones_like(inds).type(hist.dtype)
    hist = hist.scatter_add_(0,inds,ones)
    hist = hist.reshape(Nsp,Ngt)

    # -- max for each superpixel across gt segs --
    maxes = th.max(hist,1).values
    asa = th.sum(maxes)/(H*W)

    return asa.item()

def get_brbp_edges(sp,gt,r=1):

    # -- prepare --
    if not th.is_tensor(sp):
        sp = th.from_numpy(sp*1.).long()
    if not th.is_tensor(gt):
        gt = th.from_numpy(gt*1.).long()

    # -- normalize gt --
    gt = gt.long()-1

    # -- unpack --
    device = sp.device
    H,W = sp.shape

    # -- get edges --
    edges_sp = th.zeros_like(sp[:-1,:-1]).bool()
    edges_gt = th.zeros_like(gt[:-1,:-1]).bool()
    for ix in range(2):
        for jx in range(2):
            # print(ix,H-1+ix,jx,W-1+jx,sp.shape)
            edges_sp = th.logical_or(edges_sp,(sp[ix:H-1+ix,jx:W-1+jx] != sp[1:,1:]))
            edges_gt = th.logical_or(edges_gt,(gt[ix:H-1+ix,jx:W-1+jx] != gt[1:,1:]))
    Nsp_edges = th.sum(edges_sp)

    # -- fuzz the edges_sp --
    if r > 0:
        pool2d = th.nn.functional.max_pool2d
        ksize = 2*r+1
        edges_sp = edges_sp[None,None,:]*1.
        edges_sp = pool2d(edges_sp,ksize,stride=1,padding=ksize//2)
        edges_sp = edges_sp[0,0].bool()

    return edges_sp,edges_gt,Nsp_edges

def compute_bp(sp,gt,r=1):

    # -- get edges --
    edges_sp,edges_gt,Nsp_edges = get_brbp_edges(sp,gt,r)

    # -- compute number equal --
    br = th.sum(1.*edges_sp*edges_gt)/th.sum(edges_sp)
    return br.item()

def compute_br(sp,gt,r=1):

    # -- get edges --
    edges_sp,edges_gt,_ = get_brbp_edges(sp,gt,r)

    # -- compute number equal --
    br = th.sum(1.*edges_sp*edges_gt)/th.sum(edges_gt)
    return br.item()
