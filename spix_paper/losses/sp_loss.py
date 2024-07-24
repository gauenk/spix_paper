import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import stnls
from einops import rearrange
from easydict import EasyDict as edict
from spix_paper.utils import append_grid
from torch.nn.functional import one_hot

class SuperpixelLoss(nn.Module):

    def __init__(self,loss_type,compat_coeff):
        super().__init__()
        self.loss_type = loss_type
        self.compat_coeff = float(compat_coeff)
        assert self.loss_type in ["cross","mse"]

    def forward(self,labels,sims):
        assert self.loss_type in ["cross","mse"]

        # -- reshape sims --
        if sims.ndim == 5:
            sims = rearrange(sims,'b h w hs ws -> b (hs ws) (h w)')

        # -- alloc [compact loss] --
        B,F,H,W = labels.shape
        zeros = th.zeros_like(labels[:,0])

        # -- normalize across #sp for each pixel --
        # sims.shape = B, NumSuperpixels, NumPixels
        sims_nmz = sims / sims.sum(-1,keepdim=True)# (B,NumSpix,NumPix) -> (B,NP,NS)
        sims = sims.transpose(-1,-2)

        # -- prepare labels --
        labels = labels.flatten(-2,-1)
        if self.loss_type == "cross":
            labels_tgt = labels[:,0].long()
            labels = one_hot(labels_tgt)*1.
        else:
            labels = rearrange(labels,'b c hw -> b hw c')

        # -- compute "superpixel loss" --
        labels_sp = sims @ (sims_nmz @ labels)
        if self.loss_type == "cross":
            cross = torch.nn.functional.cross_entropy
            labels_sp = rearrange(labels_sp,'b hw c -> (b hw) c')
            labels_tgt = rearrange(labels_tgt,'b hw -> (b hw)')
            sp_loss = cross(th.log(labels_sp+1e-15),labels_tgt)
        elif self.loss_type == "mse":
            sp_loss = th.mean((labels - labels_sp)**2)

        # -- prepare "compact loss" --
        ixy = append_grid(zeros[:,None,None],1,normz=True)[:,0,1:]
        ix = ixy[:,0].flatten(-2,-1)
        iy = ixy[:,1].flatten(-2,-1)
        ix_sp = (sims @ (sims_nmz @ ix[:,:,None]))[...,0]
        iy_sp = (sims @ (sims_nmz @ iy[:,:,None]))[...,0]

        # -- compact --
        compact_loss_x = th.mean((ix_sp - ix)**2)
        compact_loss_y = th.mean((iy_sp - iy)**2)
        compact_loss = (compact_loss_x + compact_loss_y)/2.
        # print(compact_loss)

        # -- final loss --
        loss = sp_loss + self.compat_coeff * compact_loss
        if th.isnan(loss):
            print(sims)
            print(sp_loss)
            print(compact_loss)
            print("nan found.")
            exit()
        # exit()

        return loss

