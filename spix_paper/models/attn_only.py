"""

   AttentionOnlyDenoiser Network

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

class AttentionOnlyDenoiser(nn.Module):

    defs = dict(SuperpixelNetwork.defs)
    defs.update(SuperpixelAttention.defs)
    defs.update({"lname":"deno","net_depth":1,"conv_kernel_size":3})
    # convolution only used to improve superpixel estimation quality.

    def __init__(self, in_dim, dim, **kwargs):
        super().__init__()

        # -- unpack --
        self.net_depth = kwargs['net_depth']
        D = self.net_depth
        conv_ksize = kwargs['conv_kernel_size']
        # conv_ksize = self.unpack_conv_ksize(conv_ksize,self.net_depth)

        # # -- io layers --
        init_conv = lambda d0,d1,ksize: nn.Conv2d(d0,d1,ksize,padding="same")
        self.conv0 = init_conv(in_dim,dim,conv_ksize)
        # self.conv1 = init_conv(dim,dim,conv_ksize[0])
        # self.conv1 = init_conv(dim,in_dim,conv_ksize[-1])

        # -- learn attn scale --
        aparams = extract(kwargs,SuperpixelNetwork.defs)
        self.attn=nn.ModuleList([SuperpixelAttention(in_dim,**aparams) for _ in range(D)])

        # -- superpixel network --
        self.use_sp_net = not(kwargs['attn_type'] == "na" and kwargs['lname'] == "deno")
        spix_kwargs = extract(kwargs,SuperpixelNetwork.defs)
        self.spix_net = SuperpixelNetwork(dim,**spix_kwargs) if self.use_sp_net else None

    def forward(self, x, noise_info=None):
        """

        Forward function.

        """

        # -- unpack --
        H,W = x.shape[-2:]

        # -- first features --
        if self.use_sp_net: sims = self.spix_net(self.conv0(x))[0]
        else: sims = None

        # -- depth --
        ftrs = x
        for d in range(self.net_depth):
            ftrs = x+self.attn[d](ftrs,sims)

        # -- output --
        deno = ftrs

        return {"deno":deno,"sims":sims}
