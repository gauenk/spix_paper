"""
Neighborhood Attention 2D PyTorch Module

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

"""

import torch
import torch as th
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from einops import rearrange

from natten.functional import na2d_av, na2d_qk_with_bias
from ..models.attn_scale_net import AttnScaleNet
from stnls.search import NonLocalSearch
from stnls.agg import NonLocalGather
# from natten.functional import natten2dav, natten2dqkrpb

def run_search_fxn(q, k, kernel_size, dilation, dist_type):
    if dist_type == "prod":
        attn = na2d_qk_with_bias(q, k, None, kernel_size, dilation)
        # attn = run_stnls_search(q,k,kernel_size,dilation,dist_type)[0]
        return attn
    elif dist_type == "l2":
        attn = run_stnls_search(q,k,kernel_size,dilation,dist_type)[0]
        return attn
    else:
        raise ValueError(f"Uknown dist_type [{dist_type}]")

def run_stnls_search(q,k,kernel_size,dilation,dist_type):
    num_heads = q.shape[1]
    ws = kernel_size
    wt,ps,pt,_k = 0,1,1,-1
    stride0,stride1 = 1,1
    topk_mode = "none"
    search = NonLocalSearch(ws, wt, ps, _k, nheads=num_heads,
                            stride0=stride0, stride1=stride1,
                            dist_type=dist_type, dilation=dilation,
                            pt=pt, self_action=None, topk_mode=topk_mode,
                            reflect_bounds=True, full_ws=True,
                            use_adj=False, itype="int")
    q = rearrange(q,'b hd h w f -> b hd 1 f h w').contiguous()
    k = rearrange(k,'b hd h w f -> b hd 1 f h w').contiguous()
    attn,flows = search(q,k)
    attn = attn[:,:,0]
    flows = flows[:,:,0]
    # print("attn.shape: ",attn.shape)
    return attn,flows

def run_stnls_agg(v,attn,flows):
    # weights = th.nn.functional.softmax(10*dists,-1)
    ps,stride0 = 1,1
    agg = NonLocalGather(ps,stride0)
    attn = attn[:,:,None]
    flows = flows[:,:,None]
    v = rearrange(v,'b hd h w f -> b hd 1 f h w').contiguous()
    v = th.sum(agg(v,attn,flows),2) # b hd k t f h w
    v = rearrange(v,'b 1 1 f h w -> b f h w').contiguous()
    return v

def run_nat_agg(v,attn,ksize,dilation):
    x = na2d_av(attn, v, ksize, dilation)
    x = rearrange(x,'b 1 h w f -> b f h w')
    return x

class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            bias=False,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            proj_layer=True,
            learn_attn_scale=False,
            # detach_learn_attn=False,
            dist_type="prod",
            sp_nftrs=3):
        super().__init__()
        # assert dist_type == "prod","Only dist_type = 'prod' supported with NA"
        self.dist_type = dist_type
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        # self.detach_learn_attn = detach_learn_attn
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        assert self.rpb is None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) if proj_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        self.learn_attn_scale = learn_attn_scale
        if not self.learn_attn_scale:
            self.attn_scale_net = nn.Identity()
        else:
            self.attn_scale_net = AttnScaleNet(dim, 1, sp_nftrs)
        self.return_attn_map = False

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qk = (self.qk(x).reshape(B, H, W, 2, self.num_heads, self.head_dim)
              .permute(3, 0, 4, 1, 2, 5))
        q, k = qk[0], qk[1]
        v = (self.v(x).reshape(B, H, W, 1, self.num_heads, self.head_dim)
             .permute(3, 0, 4, 1, 2, 5))[0]
        # print("k.shape: ",k.shape)

        # -- rescaling --
        if self.learn_attn_scale:
            scale = self.attn_scale_net(rearrange(x,'b h w c -> b c h w'))
            scale = rearrange(scale,'b 1 h w -> b 1 h w 1')
            # if self.detach_learn_attn:
            #     scale = scale.detach()
            q = scale * q
        else:
            if self.dist_type == "prod":
                q = self.scale * q

        # -- attention --
        # attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = run_search_fxn(q, k, self.kernel_size, self.dilation, self.dist_type)
        if not self.learn_attn_scale and self.dist_type == "l2":
            attn = self.scale * attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b: x = x[:, :Hp, :Wp, :]

        # if self.return_attn_map:
        #     return self.proj_drop(self.proj(x)),attn
        # else:
        #     return self.proj_drop(self.proj(x))
        return self.proj_drop(self.proj(x)),attn

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )


class NeighAttnMat(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            bias=False,
            qk_bias=False,
            qk_scale=None,
            learn_attn_scale=False,
            # detach_learn_attn=False,
            dist_type="prod",
            sp_nftrs=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**(-0.5)
        # self.detach_learn_attn = detach_learn_attn
        self.dist_type = dist_type
        # assert dist_type == "prod","Only dist_type = 'prod' supported with NA"
        # print(self.scale)
        # exit()
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qk = nn.Linear(dim, dim * 2, bias=qk_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        assert self.rpb is None

        self.learn_attn_scale = learn_attn_scale
        if not self.learn_attn_scale:
            self.attn_scale_net = nn.Identity()
        else:
            self.attn_scale_net = AttnScaleNet(dim, 1, sp_nftrs)

    def forward(self, x):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        # print("x.shape: ",x.shape)
        qk = (
            self.qk(x)
            .reshape(B, H, W, 2, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k = qk[0], qk[1]

        # # -- compare --
        # diff0 = th.mean((q-x[:,None])**2).item()
        # diff1 = th.mean((k-x[:,None])**2).item()
        # print("differences: ",diff0,diff1)

        # -- rescaling --
        if self.learn_attn_scale:
            scale = self.attn_scale_net(rearrange(x,'b h w c -> b c h w'))
            scale = rearrange(scale,'b 1 h w -> b 1 h w 1')
            # if self.detach_learn_attn:
            #     # print("self.detach_learn_attn.")
            #     scale = scale.detach()
            q = scale * q
        if not self.learn_attn_scale and self.dist_type == "prod":
            q = self.scale * q
        attn = run_search_fxn(q, k, self.kernel_size, self.dilation, self.dist_type)
        if not self.learn_attn_scale and self.dist_type == "l2":
            attn = self.scale * attn
        # attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        return attn

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )


class NeighAttnAgg(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            kernel_size,
            dilation=1,
            v_bias=False,
            proj_layer=True,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()

        # -- for padding --
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.v = nn.Linear(dim, dim * 1, bias=v_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) if proj_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn):
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = pad_t = 0
            pad_r = max(0, self.window_size - W)
            pad_b = max(0, self.window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        v = (self.v(x).reshape(B, H, W, 1, self.num_heads, self.head_dim)
             .permute(3, 0, 4, 1, 2, 5))[0]
        # v[...] = 0
        # v[...,0] = 1
        # # attn = attn.softmax(dim=-1)
        # print("[nat_spin] attn.shape: ",attn.shape,v.shape)
        # print("v.shape: ",v.shape)
        # print("att.shape, v.shape: ",attn.shape,v.shape)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, self.dilation)
        # x = natten2dav(attn, v, self.kernel_size, self.dilation)
        # x = natten2dav(attn, v, self.dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
        )

def natten_padding(x,kernel_size):
    window_size = kernel_size*kernel_size
    B, Hp, Wp, C = x.shape
    H, W = int(Hp), int(Wp)
    pad_l = pad_t = pad_r = pad_b = 0
    if H < window_size or W < window_size:
        pad_l = pad_t = 0
        pad_r = max(0, window_size - W)
        pad_b = max(0, window_size - H)
        x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H, W, _ = x.shape
    pad_info = {"Hp":Hp,"Wp":Wp,"pad_r":pad_r,"pad_b":pad_b}
    return x,pad_info

def natten_remove_padding(x,pad_info):
    Hp,Wp = pad_info["Hp"],pad_info["Wp"]
    pad_r,pad_b = pad_info["pad_r"],pad_info["pad_b"]
    if pad_r or pad_b:
        x = x[:, :Hp, :Wp, :]
    return x
