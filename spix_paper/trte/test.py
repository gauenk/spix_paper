
# -- basic --
import gc
import glob,os

from scipy.io import loadmat
from skimage.segmentation import mark_boundaries

import torch
import torch as th
import numpy as np
from torchvision.utils import save_image

from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

# -- load models --
from spix_paper.models import load_model as _load_model

# -- project imports --
import spix_paper
from spix_paper import metrics
from spix_paper.data import load_data
from spix_paper.losses import load_loss
from spix_paper.models import load_model
import spix_paper.trte_utils as utils
import spix_paper.utils as base_utils

# SAVE_ROOT = Path("output/eval_superpixels/")

def run(cfg):

    # -- init experiment --
    defs = {"data_path":"./data/","data_augment":False,
            "patch_size":128,"data_repeat":1,"colors":3,
            "use_connected":False,"save_output":False,"seed":0,
            "num_samples":0,"load_checkpoint":True}
    cfg = base_utils.extract_defaults(cfg,defs)
    device = "cuda"
    save_root = Path(cfg.save_root) / cfg.tr_uuid

    # -- seed --
    base_utils.seed_everything(cfg.seed)

    # -- dataset --
    cfg.data_load_test = True
    dset,dataloader = load_data(cfg)

    # -- noise function --
    sigma = base_utils.optional(cfg,'sigma',0.)
    add_noise = lambda x: x + (sigma/255.)*th.randn_like(x)

    # -- load model --
    model = load_model(cfg)

    # -- restore from checkpoint --
    chkpt_root = utils.get_ckpt_root(cfg.tr_uuid,cfg.base_path)
    chkpt,chkpt_fn = utils.get_checkpoint(chkpt_root)

    # -- checking --
    no_chkpt = not(chkpt is None)
    is_empty = "empty" in cfg.mname
    # print(no_chkpt,is_empty,cfg.mname)
    assert no_chkpt or is_empty,"Must have a checkpoint loaded for testing."

    # -- load --
    if cfg.load_checkpoint:
        loaded_epoch = utils.load_checkpoint(chkpt,model)
        print("Restoring State from [%s]" % chkpt_fn)

    # -- to device --
    model = model.to(device)
    model = model.eval()

    # -- init info --
    ifields = ["asa","br","bp","nsp_og","nsp","hw","name",
               "pooled_psnr","pooled_ssim","entropy",
               "deno_psnr","deno_ssim"]
    info = edict()
    for f in ifields: info[f] = []

    # -- each sample --
    for ix,(img,seg) in enumerate(dataloader):

        # -- garbage collect --
        gc.collect()
        th.cuda.empty_cache()

        # -- unpack --
        img, seg = img.to(device)/255., seg.to(device)
        img, seg = img[:,:,:-1,:-1], seg[:,:-1,:-1]
        name = dset.names[ix]
        B,F,H,W = img.shape
        assert B == 1,"Testing metrics are not currently batch-able."
        # print(img.shape,seg.shape)

        # -- optional noise --
        noisy = add_noise(img)

        # -- compute superpixels --
        with th.no_grad():
            output = model(noisy)

        # -- unpack --
        deno = base_utils.optional(output,'deno',None)
        sims = base_utils.optional(output,'sims',None)
        if not(sims is None) and (sims.ndim == 5):
            sims = rearrange(sims,'b h w sh sw -> b (sh sw) (h w)')

        # -- get superpixel --
        if not(sims is None):
            spix = sims.argmax(1).reshape(B,H,W)
            spix_og = spix.clone()
            if cfg.use_connected:
                if th.is_tensor(spix): spix = spix.cpu().numpy().astype(np.int64)
                else: spix = spix.astype(np.int64)
                cmin = cfg.connected_min
                cmax = cfg.connected_max
                spix = connected_sp(spix,cmin,cmax)

            # -- spix pooled --
            # _img = img[0].cpu().numpy().transpose(1,2,0)
            # print("img.shape: ",img.shape)
            entropy = th.mean(-sims*th.log(sims+1e-15)).item()
            pooled = spix_paper.spix_utils.sp_pool(img,sims)

        else:
            spix = th.zeros_like(img[:,0]).long()
            spix_og = spix.clone()
            entropy = -1.
            pooled = th.zeros_like(img)

        # -- save --
        if cfg.save_output:
            # save_dir = save_root / ("s_%d"%cfg.S) / cfg.method
            save_dir = save_root / "spix"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            _img = img[0].cpu().numpy().transpose(1,2,0)
            viz = mark_boundaries(_img,spix,mode="subpixel")
            viz = th.from_numpy(viz.transpose(2,0,1))
            save_fn = save_dir / ("%s.jpg"%name)
            save_image(viz,save_fn)


        # -- save --
        if cfg.save_output:
            # save_dir = save_root / ("s_%d"%cfg.S) / cfg.method
            save_dir = save_root / "pooled"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            save_fn = save_root / ("%s_s.jpg"%name)
            print("Saved @ [%s]"%save_fn)
            save_image(smoothed,save_fn)

        # -- eval deno --
        deno_psnr = metrics.compute_psnrs(img,deno,div=1.).item()
        deno_ssim = metrics.compute_ssims(img,deno,div=1.).item()

        # -- eval pool --
        pooled_psnr = metrics.compute_psnrs(img,pooled,div=1.).item()
        pooled_ssim = metrics.compute_ssims(img,pooled,div=1.).item()

        # -- eval & collect info --
        iinfo = edict()
        for f in ifields: iinfo[f] = []
        iinfo.asa = metrics.compute_asa(spix[0],seg[0])
        iinfo.br = metrics.compute_br(spix[0],seg[0],r=1)
        iinfo.bp = metrics.compute_bp(spix[0],seg[0],r=1)
        iinfo.nsp = int(len(th.unique(spix)))
        iinfo.nsp_og = int(len(th.unique(spix_og)))
        iinfo.deno_psnr = deno_psnr
        iinfo.deno_ssim = deno_ssim
        iinfo.pooled_psnr = pooled_psnr
        iinfo.pooled_ssim = pooled_ssim
        iinfo.entropy = entropy
        iinfo.hw = img.shape[-2]*img.shape[-1]
        iinfo.name = name
        print(iinfo)
        for f in ifields: info[f].append(iinfo[f])
        if cfg.num_samples > 0 and ix >= cfg.num_samples:
            break

    return info
