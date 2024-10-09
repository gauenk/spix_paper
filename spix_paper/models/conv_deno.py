"""

   Conv Denoising Network

"""

# -- import torch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# -- basic --
import math
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- submodules --
from ..utils import extract
from .sp_net import SuperpixelNetwork
from ..attn import SuperpixelAttention

class ConvDenoiser(nn.Module):

    defs = dict(SuperpixelNetwork.defs)
    defs.update(SuperpixelAttention.defs)
    defs.update({"lname":"deno","net_depth":1,"conv_kernel_size":3})

    def __init__(self, in_dim, dim, **kwargs):
        super().__init__()

        # -- unpack --
        self.net_depth = kwargs['net_depth']
        D = self.net_depth
        conv_ksize = kwargs['conv_kernel_size']
        conv_ksize = self.unpack_conv_ksize(conv_ksize,self.net_depth)

        # -- io layers --
        init_conv = lambda d0,d1,ksize: nn.Conv2d(d0,d1,ksize,padding="same")
        self.conv0 = init_conv(in_dim,dim,conv_ksize[0])
        self.conv1 = init_conv(dim,in_dim,conv_ksize[-1])

        # -- learn attn scale --
        self.mid = nn.ModuleList([init_conv(dim,dim,conv_ksize[d+1]) for d in range(D-1)])
        akwargs = extract(kwargs,SuperpixelNetwork.defs)
        self.attn = nn.ModuleList([SuperpixelAttention(dim,**akwargs) for _ in range(D)])

        # -- superpixel network --
        self.use_sp_net = not(kwargs['attn_type'] == "na" and kwargs['lname'] == "deno")
        spix_kwargs = extract(kwargs,SuperpixelNetwork.defs)
        self.spix_net = SuperpixelNetwork(dim,**spix_kwargs) if self.use_sp_net else None

    def unpack_conv_ksize(self,ksize,depth):
        if hasattr(ksize,"__len__"):
            if len(ksize) == 1:
                ksize = ksize*(depth+1)
            else:
                assert len(ksize) == (depth+1),"Must be equal."
                return ksize
        else:
            return [ksize,]*(depth+1)

    def forward(self, x, noise_info=None):
        """

        Forward function.

        """

        # -- unpack --
        H,W = x.shape[-2:]

        # -- first features --
        ftrs = self.conv0(x)
        if self.use_sp_net: sims = self.spix_net(ftrs)[0]
        else: sims = None

        # -- depth --
        if self.net_depth >=1 :
            ftrs = ftrs+self.attn[0](ftrs,sims)
        for d in range(self.net_depth-1):
            ftrs = self.mid[d](ftrs)
            ftrs = ftrs+self.attn[d+1](ftrs,sims)

        # -- output --
        deno = x + self.conv1(ftrs)

        return {"deno":deno,"sims":sims}
