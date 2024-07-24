# -- pytorch --
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

# -- basics --
import math
from einops import rearrange

# -- slic iteration helper fxn --
from ..pwd import PairwiseDistFunction
from ..utils import append_grid,add_grid
from .slic_utils import init_centroid,get_abs_indices


def run_slic(pix_ftrs, stoken_size=[16, 16],
             n_iter=2, M = 0., sm_scale=1.,grad_type="full"):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6
    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        nsp: int
            A number of superpixels
        n_iter: int
            A number of iterations
        return_hard_label: bool
            return hard assignment or not
    """

    # -- unpack --
    B = pix_ftrs.shape[0]
    height, width = pix_ftrs.shape[-2:]
    if not hasattr(stoken_size,"__len__"):
        stoken_size = [stoken_size,stoken_size]
    sheight, swidth = stoken_size[0],stoken_size[1]
    nsp_height = height // sheight
    nsp_width = width // swidth
    nsp = nsp_height * nsp_width
    full_grad = grad_type == "full"

    # -- add grid --
    if th.is_tensor(M): M = M[:,None]
    pix_ftrs = append_grid(pix_ftrs[:,None],M/stoken_size[0],normz=True)[:,0]
    shape = pix_ftrs.shape

    # -- init centroids/inds --
    sftrs, ilabel = init_centroid(pix_ftrs, nsp_width, nsp_height)
    abs_indices = get_abs_indices(ilabel, nsp_width)
    mask = (abs_indices[1] >= 0) * (abs_indices[1] < nsp)
    pix_ftrs = pix_ftrs.reshape(*pix_ftrs.shape[:2], -1)
    permuted_pix_ftrs = pix_ftrs.permute(0, 2, 1).contiguous()
    coo_inds = abs_indices[:,mask]
    # print(coo_inds.shape,mask.shape,permuted_pix_ftrs.shape)

    # -- determine grad --
    with torch.set_grad_enabled(full_grad):
        for k in range(n_iter):

            # # -- compute all affinities  --
            # pwd_fxn = PairwiseDistFunction.apply
            # dist_matrix = pwd_fxn(pix_ftrs, sftrs, ilabel, nsp_width, nsp_height)
            # print(dist_matrix)
            # print(dist_matrix.std())

            # # -- sample only relevant affinity --
            # sparse_sims = (-sm_scale*dist_matrix).softmax(1)
            # reshaped_sparse_sims = sparse_sims.reshape(-1)
            # sparse_sims = torch.sparse_coo_tensor(abs_indices[:,mask],
            #                                       reshaped_sparse_sims[mask])
            # sims = sparse_sims.to_dense().contiguous()

            # -- compute all affinities  --
            sparse_sims, sims = _update_sims(pix_ftrs,sftrs,ilabel,
                                             nsp_width,nsp_height,
                                             sm_scale,coo_inds,mask)
            # -- update centroids --
            if k < n_iter - 1:
                sftrs = _update_sftrs(sims,permuted_pix_ftrs)

    # -- manage gradient; always differentiable --
    if "fixed" in grad_type:
        if grad_type == "fixed_spix":
            # -- version 0 --
            # inds = sims.view(*sims.shape[:-2],-1).argmax(-1)
            # binary = one_hot(inds,sH*sW)
            # _sims = binary.view(sims.shape).type(sims.dtype)

            # -- version 1 --
            spix = sims.argmax(1).reshape(B,-1)
            B,NSP,NP = sims.shape
            batch_inds = th.arange(B).unsqueeze(-1)
            pix_inds = th.arange(NP).unsqueeze(0)
            _sims = th.zeros_like(sims)
            _sims[batch_inds,spix,pix_inds] = 1.
        else:
            _sims = sims.detach()
        sftrs = _update_sftrs(_sims,permuted_pix_ftrs)
        sparse_sims,sims = _update_sims(pix_ftrs,sftrs,ilabel,nsp_width,
                                        nsp_height,sm_scale,coo_inds,mask)

    return sparse_sims, sims, nsp, sftrs

# def sims_to_spix_mat(sims):
#     spix = sims.argmax(1).reshape(B,-1)
#     B,NSP,NP = sims.shape
#     batch_inds = th.arange(B).unsqueeze(-1)
#     pix_inds = th.arange(NP).unsqueeze(0)
#     _sims = th.zeros_like(sims)
#     _sims[batch_inds,spix,pix_inds] = 1.
#     return _sims

# def diff_sims(sims):
#     sftrs = _update_sftrs(_sims,permuted_pix_ftrs)
#     sparse_sims,sims = _update_sims(pix_ftrs,sftrs,ilabel,nsp_width,
#                                     nsp_height,sm_scale,coo_inds,mask)
#     return sims

def _update_sftrs(sims,permuted_pix_ftrs):
    sftrs = torch.bmm(sims, permuted_pix_ftrs)/(sims.sum(2, keepdim=True) + 1e-16)
    sftrs = sftrs.permute(0, 2, 1).contiguous()
    return sftrs

def _update_sims(pix,sftrs,ilabel,nsp_width,nsp_height,sm_scale,coo_inds,mask):

    # -- compute all affinities  --
    pwd_fxn = PairwiseDistFunction.apply
    dist_matrix = pwd_fxn(pix, sftrs, ilabel, nsp_width, nsp_height)

    # -- sample only relevant affinity --
    sparse_sims = (-sm_scale*dist_matrix).softmax(1)
    reshaped_sims = sparse_sims.reshape(-1)
    sparse_sims = torch.sparse_coo_tensor(coo_inds,reshaped_sims[mask])
    sims = sparse_sims.to_dense().contiguous()
    return sparse_sims, sims

def compute_slic_params(pix_ftrs, sims, stoken_size=[16, 16], M = 0., sm_scale=1.):
    """

    With overhead

    """

    # -----------------------
    #
    #      Core Function
    #
    # -----------------------

    # -- rehsape --
    if sims.ndim == 5:
        sims = rearrange(sims,'b h w sh sw -> b (sh sw) (h w)')

    # -- unpack --
    height, width = pix_ftrs.shape[-2:]
    if not hasattr(stoken_size,"__len__"):
        stoken_size = [stoken_size,stoken_size]
    sheight, swidth = stoken_size[0],stoken_size[1]
    nsp_height = height // sheight
    nsp_width = width // swidth
    nsp = nsp_height * nsp_width

    # -- add grid --
    if th.is_tensor(M): M = M[:,None]
    pix_ftrs = append_grid(pix_ftrs[:,None],M/stoken_size[0])[:,0]
    shape = pix_ftrs.shape

    # -- init centroids/inds --
    _, ilabel = init_centroid(pix_ftrs, nsp_width, nsp_height)
    # print("ilabel.shape: ",ilabel.shape)
    abs_indices = get_abs_indices(ilabel, nsp_width)
    mask = (abs_indices[1] >= 0) * (abs_indices[1] < nsp)
    pix_ftrs = pix_ftrs.reshape(*pix_ftrs.shape[:2], -1)
    permuted_pix_ftrs = pix_ftrs.permute(0, 2, 1).contiguous()
    coo_inds = abs_indices[:,mask]

    # -----------------------
    #
    #      Core Function
    #
    # -----------------------

    # -- compute means --
    sftrs = torch.bmm(sims, permuted_pix_ftrs) / (sims.sum(2, keepdim=True) + 1e-16)
    sftrs = sftrs.permute(0, 2, 1).contiguous()

    # -- compute sims --
    _,sims = _update_sims(pix_ftrs,sftrs,ilabel,nsp_width,nsp_height,
                          sm_scale,coo_inds,mask)

    return sftrs,sims

def get_indices(B,H,W,sH,sW,device):
    sHW = sH*sW
    labels = th.arange(sHW, device=device).reshape(1, 1, sH, sW).float()
    interp = th.nn.functional.interpolate
    labels = interp(labels, size=(H, W), mode="nearest").long()
    labels = labels.repeat(B, 1, 1, 1)
    labels = labels.reshape(B, -1)
    return labels

def sparse_to_full(sims,S):
    B,_,H,W = sims.shape
    sH,sW = H//S,W//S
    sHW = sH*sW
    ilabels = get_indices(B,H,W,sH,sW,sims.device)
    abs_indices = get_abs_indices(ilabels, sW)
    mask = (abs_indices[1] >= 0) * (abs_indices[1] < sHW)
    reshaped_sparse_sims = sims.reshape(-1)
    sparse_sims = torch.sparse_coo_tensor(abs_indices[:, mask],
                                                  reshaped_sparse_sims[mask])
    sims = sparse_sims.to_dense().contiguous()
    return sims

def get_dists(img,ftrs,stride,M):
    # -- imports --
    from st_spix.utils import append_grid

    # -- set-up --
    H,W = img.shape[-2:]
    sW,sH = W//stride,H//stride
    # print(H,W,sW,sH,stride,M)
    num_spix = sW*sH
    _, lmap = calc_init_centroid(img, sW, sH)
    abs_inds = get_abs_indices(lmap, sW)
    mask = (abs_inds[1] >= 0) * (abs_inds[1] < num_spix)
    img = append_grid(img[:,None],M/stride)[:,0]

    # -- reshape --
    img = img.reshape(*img.shape[:2], -1)

    # -- compute pwd --
    pwd_fxn = PairwiseDistFunction.apply
    dists = pwd_fxn(img, ftrs, lmap, sW, sH)

    return dists

def get_lmap(img,stride):
    H,W = img.shape[-2:]
    sW,sH = W//stride,H//stride
    num_spix = sW*sH
    _, lmap = calc_init_centroid(img, sW, sH)
    return lmap

def explicit_pwd_bwd(dmat_grad,pixel,spixel,stride):
    B,F,H,W = pixel.shape
    sW,sH = W//stride,H//stride
    lmap = get_lmap(pixel,stride)
    pixel = pixel.flatten(2)
    spixel = spixel.flatten(2)
    import superpixel_cuda
    # dmat_grad = None
    pixel_grad = torch.zeros_like(pixel)
    spixel_grad = torch.zeros_like(spixel)
    pwd_bwd = superpixel_cuda.pwd_backward
    pixel_grad,spixel_grad = pwd_bwd(
        dmat_grad.contiguous(), pixel.contiguous(),
        spixel.contiguous(), lmap.contiguous(),
        pixel_grad, spixel_grad, sW, sW)
    pixel_grad = pixel_grad.reshape(B,F,H,W)
    # print(pixel_grad[0,:,128:130,128:130])
    return pixel_grad

def expand_dists(dists,img,stride):

    # -- set-up --
    H,W = img.shape[-2:]
    sW,sH = W//stride,H//stride
    # print(H,W,sW,sH,stride,M)
    num_spix = sW*sH
    _, lmap = calc_init_centroid(img, sW, sH)
    abs_inds = get_abs_indices(lmap, sW)
    mask = (abs_inds[1] >= 0) * (abs_inds[1] < num_spix)

    # -- sample only relevant affinity --
    dists = th.where(dists>1e10,0,dists)
    dists = dists.reshape(-1)
    # print("a: ",dists.shape,abs_inds.shape,mask.shape)
    dists = th.sparse_coo_tensor(abs_inds[:, mask], dists[mask])
    dists = dists.to_dense().contiguous()

    # -- reshape to match sims --
    shape_str = 'b (sh sw) (h w) -> b h w sh sw'
    dists = rearrange(dists,shape_str,h=H,sh=sH)

    return dists

def dists_to_sims(dists,H,W,scale,stride,expand=False):

    # -- features --
    B = dists.shape[0]
    sW,sH = W//stride,H//stride
    sHW = sH*sW

    # -- init centroids/inds --
    pix_ftrs = th.zeros((B,1,H,W),device=dists.device)
    _, lmap = calc_init_centroid(pix_ftrs, sW, sH)
    abs_inds = get_abs_indices(lmap, sW)
    mask = (abs_inds[1] >= 0) * (abs_inds[1] < sHW)

    # -- softmax --
    sims = (-scale*dists).softmax(1)

    # dists = dists.reshape(-1)
    # print("a: ",dists.shape,abs_inds.shape,mask.shape)
    # dists = th.sparse_coo_tensor(abs_inds[:, mask], dists[mask])
    # print("b: ",amat.shape,abs_inds.shape,mask.shape)
    # print("dists_to_sims: ",dists.shape,abs_inds.shape,rs_amat.shape,mask.shape)

    if expand:
        # -- sample only valid coordinates --
        sims = sims.reshape(-1)
        sims = torch.sparse_coo_tensor(abs_inds[:,mask],sims[mask])
        sims = sims.to_dense().contiguous()

        # -- reshape --
        shape_str = 'b (sh sw) (h w) -> b h w sh sw'
        sims = rearrange(sims,shape_str,h=H,sh=sH)

    return sims

def get_snn_pool(pix_ftrs, S=14):
    # -- init centroids/inds --
    B,F,H,W = pix_ftrs.shape
    sH,sW = H//S, W//S
    centroids,_ = calc_init_centroid(pix_ftrs, sW, sH)
    centroids = centroids.reshape(B,F,sH,sW)
    return centroids
