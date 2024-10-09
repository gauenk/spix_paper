
# -- basic --
import gc
import glob,os
import pandas as pd

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

# -- helper --
from dev_basics.net_chunks.space import run_simple_spatial_chunks

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
    save_root = Path(cfg.save_root) / cfg.tr_uuid / cfg.dname

    # -- seed --
    base_utils.seed_everything(cfg.seed)

    # -- dataset --
    cfg.data_load_test = True
    dset,dataloader = load_data(cfg)

    # -- noise function --
    sigma = base_utils.optional(cfg,'sigma',0.)
    ntype = base_utils.optional(cfg,'noise_type',"gaussian")
    def pre_process(x):
        if ntype == "gaussian":
            return x + (sigma/255.)*th.randn_like(x),{}
        elif ntype == "isp":
            return isp.run_unprocess(x)
        else:
            raise ValueError(f"Uknown noise type [{ntype}]")
    def post_process(deno,noise_info):
        if ntype == "gaussian":
            return deno
        elif ntype == "isp":
            keys = ['red_gain','blue_gain','cam2rgb']
            args = [noise_info[k] for k in keys]
            return isp.run_process(deno,*args)
        else:
            raise ValueError(f"Uknown noise type [{ntype}]")


    # -- load model --
    model = load_model(cfg)
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('#Params : {:<.4f} [K]'.format(num_parameters / 10 ** 3))

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

    # -- [dev testing] load --
    # fn = "/home/gauenk/Documents/packages/superpixel_paper/output/deno/train/checkpoints/b3a1aa3d-720d-4d8d-9de5-5c9367028735/b3a1aa3d-720d-4d8d-9de5-5c9367028735-epoch=199.ckpt"
    # utils.load_old_model(fn,model)

    # -- to device --
    model = model.to(device)
    model = model.eval()

    # -- view save dir of enabled --
    if cfg.save_output:
        print(f"Saving to root [{save_root}]")

    # -- init info --
    ifields = ["asa","br","bp","nsp_og","nsp","hw","image_name",
               "pooled_psnr","pooled_ssim","entropy",
               "deno_psnr","deno_ssim","ycbcr_psnr","ycbcr_ssim"]
    info = edict()
    for f in ifields: info[f] = []

    # -- each sample --
    for ix,(img,seg) in enumerate(dataloader):

        # -- garbage collect --
        gc.collect()
        th.cuda.empty_cache()

        # -- unpack --
        img, seg = img.to(device)/255., seg.to(device)
        print(img.shape,seg.shape)
        # if cfg.dname == "manga109": # 1170,826 -> 1168,824
        #     img = img[:,:,1:-1,1:-1]
        if cfg.dname == "bsd500":
            img,seg = img[...,:-1,:-1],seg[...,:-1,:-1]
        name = dset.names[ix]
        B,F,H,W = img.shape
        assert B == 1,"Testing metrics are not currently batch-able."
        # print(img.shape,seg.shape)

        # -- optional noise --
        th.manual_seed(cfg.seed+ix)
        noisy,ninfo = pre_process(img)

        # -- compute superpixels --
        with th.no_grad():
            output = crop_test(model,noisy)

        # -- unpack --
        deno = base_utils.optional(output,'deno',None)
        sims = base_utils.optional(output,'sims',None)

        # # -- [dev] --
        # save_image(th.clamp(deno,0.,1.),f"new_{name}.png")

        # -- optionally post-process --
        deno = post_process(deno,ninfo)
        if not(sims is None) and (sims.ndim == 5):
            sims = rearrange(sims,'b h w sh sw -> b (sh sw) (h w)')
        deno = th.clamp(deno,0.,1.)

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
            save_dir = save_root / "deno"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            clipped = th.clamp(deno,0.,1.)
            save_fn = save_dir / ("%s.jpg"%name)
            save_image(clipped,save_fn)

            save_dir = save_root / "res"
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            res = (clipped - img).abs()
            res = res - res.min()
            res = res / (res.max()+1e-10)
            save_fn = save_dir / ("%s.jpg"%name)
            save_image(res,save_fn)

        # -- save --
        if cfg.save_output and not(sims is None):
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
        if cfg.save_output and not(sims is None):
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

        # -- eval at ycbr --
        img_c = th.clamp(img*255.,0.,255.)
        deno_c = th.clamp(deno*255.,0.,255.)
        rgb_to_ycbcr = base_utils.rgb_to_ycbcr
        img_ycbcr,deno_ycbcr = rgb_to_ycbcr(img_c/255.),rgb_to_ycbcr(deno_c/255.)
        ycbcr_psnr = metrics.compute_psnrs(img_ycbcr[:,:1],deno_ycbcr[:,:1],div=255.).item()
        ycbcr_ssim = metrics.compute_ssims(img_ycbcr[:,:1],deno_ycbcr[:,:1],div=255.).item()

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
        iinfo.ycbcr_psnr = ycbcr_psnr
        iinfo.ycbcr_ssim = ycbcr_ssim
        iinfo.pooled_psnr = pooled_psnr
        iinfo.pooled_ssim = pooled_ssim
        iinfo.entropy = entropy
        iinfo.hw = img.shape[-2]*img.shape[-1]
        iinfo.image_name = name
        # print(iinfo)
        for f in ifields: info[f].append(iinfo[f])
        if cfg.num_samples > 0 and ix >= cfg.num_samples:
            break

    print(pd.DataFrame(info)[['image_name','deno_psnr','deno_ssim']])
    # exit()
    return info

def crop_test(model,img,cropsize=256,overlap=0.25):
    fwd_fxn = lambda x: model(x)['deno']
    deno = run_simple_spatial_chunks(fwd_fxn,img,cropsize,overlap)
    return {"deno":deno,"sims":None}
