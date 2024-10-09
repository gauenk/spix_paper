
# -- basic imports --
import torch
import torch as th
from einops.layers.torch import Rearrange

import torch
import torch as th
from torch import nn
from torch.nn.functional import pad,one_hot
from torch.nn.init import trunc_normal_
from einops import rearrange,repeat

# -- basic utils --
from spix_paper.utils import extract_self

# -- attn modules --
from .nat import NeighAttnMat
from .nat import NeighAttnAgg
from .attn_reweight import AttnReweight
from ..spix_utils import compute_slic_params
from ..utils import get_fxn_kwargs

class SuperpixelAttention(nn.Module):
    """

    Superpixel Attention Module

    """

    defs = {"attn_type":"soft","dist_type":"prod","normz_patch":False,
            "qk_scale":None,"nheads":1,"kernel_size":5,"dilation":1,
            "use_proj":True,"use_weights":True,"learn_attn_scale":False,
            "attn_drop":0.0,"proj_drop":0.0,"detach_sims":False,
            "qk_layer":True,"v_layer":True,"proj_layer":True,
            "sp_nftrs":None,"proj_attn_layer":False,
            "proj_attn_bias":False,"run_attn_search":True}

    def __init__(self,dim,**kwargs):
        super().__init__()

        # -- init --
        extract_self(self,kwargs,self.defs)

        # -- check superpixels --
        assert self.attn_type in ["soft","hard","hard+grad","na"]

        # -- attention modules --
        nheads = self.nheads#kwargs['nheads']
        kernel_size = self.kernel_size#kwargs['kernel_size']
        kwargs_attn = get_fxn_kwargs(kwargs,NeighAttnMat.__init__)
        self.nat_attn = NeighAttnMat(dim,nheads,kernel_size,**kwargs_attn)
        self.attn_rw = AttnReweight()
        kwargs_agg = get_fxn_kwargs(kwargs,NeighAttnAgg.__init__)
        self.nat_agg = NeighAttnAgg(dim,nheads,kernel_size,**kwargs_agg)

        # -- [optional] project attention to create a weighted conv --
        assert self.run_attn_search or self.proj_attn_layer,"At least one must be true."
        if self.proj_attn_layer:
            bias = self.proj_attn_bias
            dim = kernel_size * kernel_size
            self.proj_attn = nn.Linear(dim,dim,bias=bias)
        else:
            self.proj_attn = nn.Identity()

    def attn_post_process(self,attn):
        if not (self.run_attn_search is True):
            attn = 0*attn
        attn = self.proj_attn(attn)
        return attn

    def forward(self, x, sims, state=None):

        # -- unpack superpixel info --
        if self.detach_sims:
            sims = sims.detach()

        # -- compute attn differences  --
        x = x.permute(0,2,3,1) # b f h w -> b h w f
        attn = self.nat_attn(x)
        attn = self.attn_post_process(attn)
        attn = self.attn_sign(attn)

        # -- reweight attn map --
        if self.attn_type == "hard":
            sH,sW = sims.shape[-2:]
            inds = sims.view(*sims.shape[:-2],-1).argmax(-1)
            binary = one_hot(inds,sH*sW).reshape_as(sims).type(sims.dtype)
            attn = self.attn_rw(attn,binary,self.normz_patch)
        elif self.attn_type in ["soft","hard+grad"]:
            sims = sims.contiguous()
            attn = self.attn_rw(attn,sims,self.normz_patch)
        elif self.attn_type == "na":
            attn = attn.softmax(-1)
        else:
            raise ValueError(f"Uknown self.attn_type [{self.attn_type}]")

        # -- aggregate --
        x = self.nat_agg(x,attn)

        # -- prepare --
        x = x.permute(0,3,1,2)#.clone() # b h w f -> b f h w
        return x

    def attn_sign(self,attn):
        # print("self.dist_type: ",self.dist_type)
        if self.dist_type == "prod":
            return attn
        elif self.dist_type == "l2":
            return -attn
        else:
            raise ValueError(f"Uknown dist type [{self.dist_type}]")

