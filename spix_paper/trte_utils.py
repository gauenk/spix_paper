import os
import sys
import math
import yaml
from math import exp
import datetime
import importlib
import numpy as np

import glob
from pathlib import Path
from easydict import EasyDict as edict
from einops import rearrange,repeat
from spix_paper import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# -- dnd dataset --
import scipy.io as sio


def get_ckpt_root(uuid,base_path,resume_uuid=None,resume_flag=True):
    resume_uuid = uuid if resume_uuid is None else resume_uuid
    resume_epoch = -1
    if resume_flag:
        dir0 = Path(base_path) / "checkpoints" / resume_uuid
        dir1 = Path(base_path) / "checkpoints" / uuid
        epoch0 = epoch_from_chkpt(dir0)
        epoch1 = epoch_from_chkpt(dir1)
        if epoch0 >= epoch1:
            root = dir0
            resume_epoch = epoch0
        else:
            root = dir1
            resume_epoch = epoch1
    else:
        root = None
        resume_epoch = -1
    if (not(root is None) and not(root.exists())) or (resume_epoch < 0):
        root = None
    return root

def get_checkpoint(root):
    if root is None: return None,""
    chkpt_files = glob.glob(os.path.join(root, "*.ckpt"))
    if len(chkpt_files) == 0: return None,""
    chkpt_files = sorted(chkpt_files, key=lambda
                         x: int(x.replace('.ckpt','').split('=')[-1]))
    chkpt = torch.load(chkpt_files[-1])
    return chkpt,chkpt_files[-1]

def load_old_model(fn,model):

    # -- init load --
    old_state = torch.load(fn)['model_state_dict']
    N=len("module.")
    old_state = {k[N:]:v for k,v in old_state.items()}

    # -- pairs --
    pairs = {"first_conv.weight":"lin0.weight",
             "first_conv.bias":"lin0.bias",
             "last_conv.weight":"lin1.weight",
             "last_conv.bias":"lin1.bias",
             "blocks.0.nat_layer.1.qk.weight":"attn.nat_attn.qk.weight",
             "blocks.0.nat_layer.1.v.weight":"attn.nat_agg.v.weight",
             "blocks.0.nat_layer.1.proj.weight":"attn.nat_agg.proj.weight",
             "blocks.0.nat_layer.1.proj.bias":"attn.nat_agg.proj.bias"}

    # -- copy weights --
    state = {}
    for old,new in pairs.items():
        state[new] = old_state[old]
        if old.startswith("first_conv"):
            state[new] = state[new].squeeze()
        if old.startswith("last_conv"):
            state[new] = state[new].squeeze()

    # -- copy attn scale net --
    # "blocks.0.nat_layer.1.attn_scale_net":"attn.nat_attn.attn_scale_net"
    for key in old_state:
        if key.startswith("blocks.0.nat_layer.1.attn_scale_net"):
            new_key = key.replace("blocks.0.nat_layer.1.attn_scale_net","attn.nat_attn.attn_scale_net")
            state[new_key] = old_state[key]

    # print(list(state.keys()))
    model.load_state_dict(state)

def load_checkpoint(chkpt,model,optimizer=None,weights_only=False,skip_module=True):
    start_epoch = 0
    if not(chkpt is None):
        prev_epoch = 0 if weights_only else chkpt['epoch']
        start_epoch = prev_epoch + 1
        state_dict = chkpt['model_state_dict']
        N=len("module.")
        if skip_module: state_dict = {k[N:]:v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)
        if not(weights_only) and not(optimizer is None):
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    return start_epoch

def init_stat_dict(chkpt,reset):
    if reset or (chkpt is None):
        stat_dict = get_stat_dict()
    else:
        stat_dict = chkpt['stat_dict']
    return stat_dict

def init_logging(cfg,experiment_name,base_path):

    # -- logging --
    log_path = os.path.join(base_path, 'logs', experiment_name)
    log_name = os.path.join(log_path,'log.txt')
    chkpt_path = os.path.join(base_path, 'checkpoints', experiment_name)

    # -- init log --
    if not os.path.exists(log_path): os.makedirs(log_path)
    if not os.path.exists(chkpt_path): os.makedirs(chkpt_path)

    # -- dump exp params --
    exp_params = vars(cfg)
    exp_params_name = os.path.join(log_path,'config.yml')
    with open(exp_params_name, 'w') as exp_params_file:
        yaml.dump(exp_params, exp_params_file, default_flow_style=False)

    return log_path,log_name,chkpt_path

def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    image = image / 255. ## image in range (0, 1)
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    cb: torch.Tensor = -37.797 * r + -74.203 * g + 112.0 * b + 128.0
    cr: torch.Tensor = 112.0 * r + -93.786 * g + -18.214 * b + 128.0

    return torch.stack((y, cb, cr), -3)

def prepare_qat(model):
    ## fuse model
    model.module.fuse_model()
    ## qconfig and qat-preparation & per-channel quantization
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    # model.qconfig = torch.quantization.QConfig(
    #     activation=torch.quantization.FakeQuantize.with_args(
    #         observer=torch.quantization.MinMaxObserver, 
    #         quant_min=-128,
    #         quant_max=127,
    #         qscheme=torch.per_tensor_symmetric,
    #         dtype=torch.qint8,
    #         reduce_range=False),
    #     weight=torch.quantization.FakeQuantize.with_args(
    #         observer=torch.quantization.MinMaxObserver, 
    #         quant_min=-128, 
    #         quant_max=+127, 
    #         dtype=torch.qint8, 
    #         qscheme=torch.per_tensor_symmetric, 
    #         reduce_range=False)
    # )
    model = torch.quantization.prepare_qat(model, inplace=True)
    return model

def import_module(name):
    return importlib.import_module(name)

def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)
    return float(psnr)

def calc_ssim(sr, hr):
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')
    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_stat_dict():
    stat_dict = {
        'epochs': 0,
        'losses': [],
        'ema_loss': 0.0,
        'set5': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'set14': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'b100': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'u100': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'manga109': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        }
    }
    return stat_dict

   
import warnings



def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def epoch_from_chkpt(ckpt_dir):
    if not Path(ckpt_dir).exists():
        return -1
    chkpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if len(chkpt_files) == 0: return -1
    chkpt_files = sorted(chkpt_files,
                         key=lambda x: int(x.replace('.ckpt','').split('=')[-1]))
    chkpt = torch.load(chkpt_files[-1])
    prev_epoch = chkpt['epoch']
    return prev_epoch

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def view_buff(buff):
    # -- print buffer --
    # print(buff['br'])
    for k,v in buff.items():
        args = np.where(~np.isnan(v))[0]
        print("%s: %2.4f" % (k,np.mean(np.array(v)[args])))

def update_agg(info,buff,epoch):
    for key in buff:
        if not(key in info): info[key] = []
        val = buff[key]
        args = np.where(~np.isnan(val))[0]
        mean = np.mean(np.array(val)[args])
        info[key].append(mean)
    if not("epoch" in info): info['epoch'] = []
    info['epoch'].append(epoch)

def init_metrics_buffer():
    pairs = {"psnr","ssim","asa","br","bp"}
    cfg = edict({k:[] for k in pairs})
    return cfg

def update_metrics_buffer(buff,img,seg,deno,sims):
    # print("img.shape: ",img.shape)
    # print("sims.shape: ",sims.shape)
    # print("seg.shape: ",seg.shape)
    if not(sims is None) and sims.ndim == 5:
        sims = rearrange(sims,'b h w sh sw -> b h w (sh sw)')
    B,F,H,W = img.shape
    psnr = metrics.compute_psnrs(img,deno,div=1.).item()
    ssim = metrics.compute_ssims(img,deno,div=1.).item()
    asa,br,bp = -1,-1,-1
    if not(sims is None):
        spix = sims.argmax(-1).reshape(B,H,W)
        asa = metrics.compute_asa(spix[0],seg[0,0])
        br = metrics.compute_br(spix[0],seg[0,0],r=1)
        bp = metrics.compute_bp(spix[0],seg[0,0],r=1)
    pairs = {"psnr":psnr,"ssim":ssim,"asa":asa,"br":br,"bp":bp}
    for k,v in pairs.items(): buff[k].append(v)
    # print("psnr: "+str(psnr))
    # print("ssim: "+str(ssim))
    # print("asa: " + str(asa))
    # print("br: " + str(br))
    # print("asa: " + str(bp))

"""

   Helper functions for DND dataset

"""

def read_dnd(noisy,bayer_pattern,info,boxes,k):
    # Crops the image to this bounding box.
    idx = [
        int(boxes[k, 0] - 1),
        int(boxes[k, 2]),
        int(boxes[k, 1] - 1),
        int(boxes[k, 3])
    ]
    noisy_crop = noisy[idx[0]:idx[1], idx[2]:idx[3]].copy()

    # Flips the raw image to ensure RGGB Bayer color pattern.
    if (bayer_pattern == [[1, 2], [2, 3]]):
      pass
    elif (bayer_pattern == [[2, 1], [3, 2]]):
      noisy_crop = np.fliplr(noisy_crop)
    elif (bayer_pattern == [[2, 3], [1, 2]]):
      noisy_crop = np.flipud(noisy_crop)
    else:
      print('Warning: assuming unknown Bayer pattern is RGGB.')

    # Loads shot and read noise factors.
    nlf_h5 = info[info['nlf'][0][i]]
    shot_noise = nlf_h5['a'][0][0]
    read_noise = nlf_h5['b'][0][0]

    # Extracts each Bayer image plane.
    denoised_crop = noisy_crop.copy()
    height, width = noisy_crop.shape
    channels = []
    for yy in range(2):
      for xx in range(2):
        noisy_crop_c = noisy_crop[yy:height:2, xx:width:2].copy()
        channels.append(noisy_crop_c)
    channels = np.stack(channels, axis=-1)
    return channels

def save_dnd(denoised_crop,output,height,width):

    # Copies denoised results to output denoised array.
    for yy in range(2):
      for xx in range(2):
        denoised_crop[yy:height:2, xx:width:2] = output[:, :, 2 * yy + xx]

    # Flips denoised image back to original Bayer color pattern.
    if (bayer_pattern == [[1, 2], [2, 3]]):
      pass
    elif (bayer_pattern == [[2, 1], [3, 2]]):
      denoised_crop = np.fliplr(denoised_crop)
    elif (bayer_pattern == [[2, 3], [1, 2]]):
      denoised_crop = np.flipud(denoised_crop)

    # Saves denoised image crop.
    denoised_crop = np.clip(np.float32(denoised_crop), 0.0, 1.0)
    save_file = os.path.join(output_dir, '%04d_%02d.mat' % (i + 1, k + 1))
    sio.savemat(save_file, {'denoised_crop': denoised_crop})


if __name__ == '__main__':
    timestamp = cur_timestamp_str()
    print(timestamp)
