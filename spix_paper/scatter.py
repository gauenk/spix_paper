import torch as th
import st_spix_cuda
from einops import rearrange,repeat

def run(img,flow,swap_c=True):
    if swap_c:
        img = rearrange(img,'b c h w -> b h w c')
        flow = rearrange(flow,'b c h w -> b h w c')
    scatter,cnts = st_spix_cuda.scatter_img_forward(img.contiguous(),
                                                    flow.contiguous())
    if swap_c:
        scatter = rearrange(scatter,'b h w c -> b c h w')
    return scatter,cnts

