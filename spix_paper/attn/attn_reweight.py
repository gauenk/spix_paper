
import torch
import torch as th

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_


import torch
from torch.autograd import Function
import spix_paper_cuda

from einops import rearrange
from .gather_sims import GatherSims

class AttnReweight(nn.Module):
    """

    Input: QK^T
    Output P(L_j=s)exp(QK^T)/[\sum_{j\in N(s)} P(L_j=s)exp(QK^T)]


    using "log(P(L_i=s))" gives us "nan" which we could ignore, but I just
    don't like having "nan"s in the pathway of my code.

    I'd rather do this without any "log P(L_i=s)" giving rows of nans.

    """

    def __init__(self):
        super().__init__()

    def forward(self, attn, sims, normz_patch=False):

        # -- get index map --
        B,H,W,sH,sW = sims.shape
        sinds = get_indices(H,W,sH,sW,sims.device)

        # -- \tilde{w}_{ij}p(s_j = s) --
        eps = 1e-15
        c = th.max(attn,dim=-1,keepdim=True).values
        attn = th.exp(attn-c)
        attn = AttnReweightFunction.apply(attn,sims,sinds)

        # -- adjust for variable nz in patch --
        if normz_patch:
            ones = th.ones_like(attn[:,:,0])
            gamma_s = AttnReweightFunction.apply(ones,sims,sinds)
            gamma_s = 1./(gamma_s.sum(2,keepdim=True)+eps)
            attn = attn * gamma_s

        # -- reweight with p(s_i = s) --
        gather_sims = GatherSims()
        pi = gather_sims(sims,sinds)
        pi = rearrange(pi,'b h w ni -> b 1 ni h w 1')
        attn = th.sum(pi * attn,2).contiguous() # sum over number of superpixels

        # # -- normalize --
        attn = attn / (eps+th.sum(attn,-1,keepdim=True)) # sum over neighbors

        return attn

    def extra_repr(self) -> str:
        return (f"attn reweight")

class AttnReweightFunction(Function):

    @staticmethod
    def forward(ctx, attn_in, sims, sinds):
        """

        Computes: P(Li = s) exp( d(qi,kj) )

        """
        NSP = 9
        dtype = attn_in.dtype
        device = attn_in.device
        B,HD,H,W,K = attn_in.shape
        attn_out = th.zeros((B,HD,NSP,H,W,K),device=device,dtype=dtype)
        spix_paper_cuda.reweight_forward(attn_out, attn_in, sims, sinds)
        ctx.save_for_backward(attn_out, attn_in, sims, sinds)
        return attn_out

    @staticmethod
    def backward(ctx, d_attn_out):
        d_attn_out = d_attn_out.contiguous()
        d_attn_in = th.zeros_like(ctx.saved_variables[1])
        d_sims = th.zeros_like(ctx.saved_variables[2])
        spix_paper_cuda.reweight_backward(
            d_attn_in,d_sims,d_attn_out,
            ctx.saved_variables[0],
            ctx.saved_variables[1],
            ctx.saved_variables[2],
            ctx.saved_variables[3],
        )

        return d_attn_in,d_sims,None


def get_indices(H,W,sH,sW,device):
    sHW = sH*sW
    labels = th.arange(sHW, device=device).reshape(1, 1, sH, sW).float()
    interp = th.nn.functional.interpolate
    labels = interp(labels, size=(H, W), mode="nearest").long()[0,0]
    labels = th.stack([labels/sW,labels%sW],-1).long()
    return labels
