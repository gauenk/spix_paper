

import copy
dcopy = copy.deepcopy
import torch
import torch as th
import numpy as np
from einops import rearrange,repeat
from skimage import io, color
from easydict import EasyDict as edict

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_gpus(ids_str): # example: [0,1,2]
    gpu_ids_str = str(gpu_ids).replace('[','').replace(']','')
    # print("gpu_ids_str: ",gpu_ids_str)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)

def add_noise(lr,args):
    if "sigma" in args:
        sigma = args.sigma
    else:
        sigma = 0.
    # print("lr[max,min]: ",lr.max().item(),lr.min().item())
    lr = lr + sigma*th.randn_like(lr)
    return lr

def extract_self(self,kwargs,defs):
    for k in defs:
        setattr(self,k,optional(kwargs,k,defs[k]))

def extract(_cfg,defs):
    return extract_defaults(_cfg,defs)

def extract_defaults(_cfg,defs):
    cfg = edict(dcopy(_cfg))
    for k in defs: cfg[k] = optional(cfg,k,defs[k])
    return cfg

def extract_defaults_new(cfg,defs):
    _cfg = edict()
    for k in defs: _cfg[k] = optional(cfg,k,defs[k])
    return _cfg

def optional(pydict,key,default):
    if pydict is None: return default
    elif key in pydict: return pydict[key]
    else: return default

def get_fxn_defaults(fxn):
    vals = fxn.__defaults__
    nargs = fxn.__code__.co_argcount
    names = fxn.__code__.co_varnames
    names = names[:nargs][-len(vals):]
    # print(fxn.__defaults__)
    # print(fxn.__code__.co_names)
    # print(fxn.__code__.co_varnames)
    return {n:v for n,v in zip(names,vals)}

def get_fxn_kwargs(cfg,fxn):
    kwargs = get_fxn_defaults(fxn)
    for key in kwargs:
        if key in cfg:
            kwargs[key] = cfg[key]
    return kwargs

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def append_grid(vid,R,normz=False):
    B,T,F,H,W = vid.shape
    dtype,device = vid.dtype,vid.device
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    if normz:
        grid_y,grid_x = grid_y/(H-1),grid_x/(W-1)
        assert grid_y.max() == 1,grid_x.max() == 1
    grid = th.stack((grid_x, grid_y), -1).float()  # 2, W(x), H(y)
    grid = repeat(grid,'h w two -> b t two h w',b=B,t=T).to(device)
    vid = th.cat([vid,R*grid],2)
    return vid

def add_grid(vid,R):
    B,T,F,H,W = vid.shape
    dtype,device = vid.dtype,vid.device
    grid_y, grid_x = th.meshgrid(th.arange(0, H, dtype=dtype, device=device),
                                 th.arange(0, W, dtype=dtype, device=device))
    grid = th.stack((grid_x, grid_y), -1).float()  # 2, W(x), H(y)
    grid = rearrange(grid,'h w two -> 1 1 two h w').to(device)
    zeros = th.zeros_like(vid[:1,:1,:-2])
    # print(grid.shape,zeros.shape,vid.shape)
    to_add = th.cat([zeros,R*grid],-3)
    # vid[:,:,-2:] = vid[:,:,-2:] + R*grid
    # vid = th.cat([vid,R*grid],2)
    vid = vid + to_add
    return vid


def rgb2lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to Lab.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The L channel values are in the range 0..100. a and b are in the range -128..127.
    """

    # Convert from sRGB to Linear RGB
    lin_rgb = torch.where(image > 0.04045,
                          torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)
    r,g,b = lin_rgb[..., 0,:,:],lin_rgb[..., 1,:,:],lin_rgb[..., 2,:,:]
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    xyz_im = torch.stack([x, y, z], -3)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883],device=xyz_im.device,
                                 dtype=xyz_im.dtype)[None, :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x,y,z = xyz_int[..., 0,:,:],xyz_int[..., 1,:,:],xyz_int[..., 2,:,:]
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    _b = 200.0 * (y - z)
    out = torch.stack([L, a, _b], dim=-3)

    return out



