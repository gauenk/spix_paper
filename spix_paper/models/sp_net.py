
# -- pytorch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad,one_hot
from torch.nn.init import trunc_normal_

# -- basics --
import math
from einops import rearrange

# -- basic utils --
from spix_paper.utils import extract_self

# -- superpixel utils --
from spix_paper.spix_utils import run_slic,sparse_to_full
from spix_paper.spix_utils import compute_slic_params


# -- superpixel --
from .ssn_net import SsnUNet
from .attn_scale_net import AttnScaleNet
from spix_paper.pwd.pair_wise_distance import PairwiseDistFunction

class SuperpixelNetwork(nn.Module):
    defs = {"sp_type":None,"sp_niters":2,"sp_m":0.,"sp_stride":8,
            "sp_scale":1.,"sp_grad_type":"full","sp_nftrs":9,"unet_sm":True,
            "attn_type":None}

    def __init__(self, dim, **kwargs):

        super().__init__()

        # -- init --
        extract_self(self,kwargs,self.defs)

        # -- check network types --
        assert self.sp_type in ["slic","slic+lrn","ssn"]
        self.use_slic = "slic" in self.sp_type
        self.use_ssn = "ssn" in self.sp_type
        self.use_lmodel = "lrn" in self.sp_type

        # -- input dimension --
        id_l0,id_l1 = nn.Identity(),nn.Identity()
        self.ssn = SsnUNet(dim,9,self.sp_nftrs,self.unet_sm) if self.use_ssn else id_l0
        self.lmodel = AttnScaleNet(dim,2,self.sp_nftrs) if self.use_lmodel else id_l1

    def _reshape_sims(self,x,sims):
        if sims.ndim != 5:
            H = x.shape[-2]
            sH = H//self._get_stride()
            shape_str = 'b (sh sw) (h w) -> b h w sh sw'
            sims = rearrange(sims,shape_str,h=H,sh=sH)
        return sims

    def _get_stride(self):
        if hasattr(self.sp_stride,"__len__"):
            return self.sp_stride[0]
        else:
            return self.sp_stride

    def forward(self, x):

        # -- unpack --
        B,F,H,W = x.shape
        sp_stride = self._get_stride()
        sH = H//sp_stride

        if self.use_slic:

            # -- use [learned or fixed] slic parameters --
            if self.use_lmodel:
                ssn_params = self.lmodel(x).reshape(x.shape[0],2,-1)
                m_est,temp_est = ssn_params[:,[0]],ssn_params[:,[1]]
                m_est = m_est.reshape((B,1,H,W))
            else:
                m_est,temp_est = self.sp_m,self.sp_scale

            # -- run slic iterations --
            # print(self.sp_stride,self.sp_niters,self.sp_grad_type)
            output = run_slic(x, self.sp_stride, self.sp_niters,
                              m_est, temp_est, self.sp_grad_type)
            s_sims, sims, num_spixels, ftrs = output
            sims = self._reshape_sims(x,sims)

        else:

            # -- directly predict slic probs from networks --
            sims = sparse_to_full(self.ssn(x),sp_stride)
            shape_str = 'b (sh sw) (h w) -> b h w sh sw'
            sims = rearrange(sims,shape_str,h=H,sh=sH)
            num_spixels, ftrs, s_sims = None, None, None

        # -- modify via attn type --
        if self.attn_type == "hard+grad":
            sH,sW = sims.shape[-2:]
            inds = sims.view(*sims.shape[:-2],-1).argmax(-1)
            binary = one_hot(inds,sH*sW)
            sims = binary.view(sims.shape).type(sims.dtype)
            sims = compute_slic_params(x, sims, self.sp_stride,
                                       self.sp_m, self.sp_scale)[1]
            shape_str = 'b (sh sw) (h w) -> b h w sh sw'
            sims = rearrange(sims,shape_str,h=H,sh=sH)

        return sims, num_spixels, ftrs, s_sims

