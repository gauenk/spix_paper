"""

   Linear Denoising Network

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

class LinearDenoiser(nn.Module):

    defs = dict(SuperpixelNetwork.defs)
    defs.update(SuperpixelAttention.defs)
    defs.update({"lname":"deno","net_depth":1})

    def __init__(self, in_dim, dim, **kwargs):
        super().__init__()

        # -- unpack --
        self.net_depth = kwargs['net_depth']
        D = self.net_depth

        # -- io layers --
        self.lin0 = nn.Linear(in_dim,dim,bias=True)
        self.lin1 = nn.Linear(dim,in_dim,bias=True)

        # -- learn attn scale --
        akwargs = extract(kwargs,SuperpixelNetwork.defs)
        self.mid = nn.ModuleList([nn.Linear(dim,dim,bias=False) for _ in range(D-1)])
        self.attn = nn.ModuleList([SuperpixelAttention(dim,**akwargs) for _ in range(D)])

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
        shape0 = lambda x: rearrange(x,'b c h w -> b (h w) c')
        shape1 = lambda x: rearrange(x,'b (h w) c -> b c h w',h=H,w=W)
        apply_lin = lambda x,lin: shape1(lin(shape0(x)))

        # -- first features --
        ftrs = apply_lin(x,self.lin0)
        if self.use_sp_net: sims = self.spix_net(ftrs)[0]
        else: sims = None

        # -- depth --
        ftrs = ftrs+self.attn[0](ftrs,sims)
        for d in range(self.net_depth-1):
            ftrs = apply_lin(ftrs,self.mid[d])
            ftrs = ftrs+self.attn[d+1](ftrs,sims)

        # -- output --
        deno = x + apply_lin(ftrs,self.lin1)

        return {"deno":deno,"sims":sims}

