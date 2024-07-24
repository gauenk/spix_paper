import torch
import torch as th

import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

import torch
from torch.autograd import Function
import spix_paper_cuda

class GatherSims(nn.Module):
    """

    Gather P(L_i = s) into a shape for pointwize mult with attn map

    """

    def __init__(self):
        super().__init__()

    def forward(self, sims, sinds):
        sims_g = GatherSimsFunction.apply(sims,sinds)
        return sims_g

    def extra_repr(self) -> str:
        return (f"attn reweight")

class GatherSimsFunction(Function):

    @staticmethod
    def forward(ctx, sims_in, sinds):
        """

        Computes: P(Li = s) exp( d(qi,kj) )

        """
        NSP = 9
        dtype = sims_in.dtype
        device = sims_in.device
        B,H,W,sH,sW =sims_in.shape
        sims_out = th.zeros((B,H,W,NSP),device=device,dtype=dtype)
        spix_paper_cuda.gather_sims_forward(sims_out, sims_in, sinds)
        ctx.save_for_backward(sims_in, sinds)
        ctx.sH = sH
        ctx.sW = sW
        return sims_out

    @staticmethod
    def backward(ctx, d_sims_out):
        B,H,W = d_sims_out.shape[:3]
        d_sims_out = d_sims_out.contiguous()
        dtype = d_sims_out.dtype
        device = d_sims_out.device
        d_sims_in = th.zeros((B,H,W,ctx.sH,ctx.sW),device=device,dtype=dtype)
        spix_paper_cuda.gather_sims_backward(d_sims_in,d_sims_out,ctx.saved_variables[1])
        return d_sims_in,None

