"""

   Simple Denoising Network

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

class SimpleDenoiser(nn.Module):

    defs = dict(SuperpixelNetwork.defs)
    defs.update(SuperpixelAttention.defs)
    defs.update({"lname":"deno"})

    def __init__(self, dim, **kwargs):
        super().__init__()

        # -- linear --
        self.lin0 = nn.Linear(3,dim,bias=False)
        self.lin1 = nn.Linear(dim,3,bias=False)

        # -- learn attn scale --
        attn_kwargs = extract(kwargs,SuperpixelNetwork.defs)
        self.attn = SuperpixelAttention(dim,**attn_kwargs)

        # -- superpixel network --
        self.use_sp_net = not(kwargs['attn_type'] == "na" and kwargs['lname'] == "deno")
        spix_kwargs = extract(kwargs,SuperpixelNetwork.defs)
        self.spix_net = SuperpixelNetwork(dim,**spix_kwargs) if self.use_sp_net else None

    def forward(self, x):
        """

        Forward function.

        """

        # -- unpack --
        H,W = x.shape[-2:]
        shape0 = lambda x: rearrange(x,'b c h w -> b (h w) c')
        shape1 = lambda x: rearrange(x,'b (h w) c -> b c h w',h=H,w=W)
        apply_lin = lambda x,lin: shape1(lin(shape0(x)))

        # -- forward --
        ftrs = apply_lin(x,self.lin0)
        if self.use_sp_net: sims = self.spix_net(ftrs)[0]
        else: sims = None
        ftrs = ftrs+self.attn(ftrs,sims)
        deno = x + apply_lin(ftrs,self.lin1)

        return {"deno":deno,"sims":sims}

