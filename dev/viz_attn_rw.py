
# -- tensor opts --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- drawing help --
import torchvision.utils as tv_utils
from skimage.segmentation import mark_boundaries
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes,draw_segmentation_masks

# -- basics --
from pathlib import Path
from easydict import EasyDict as edict

# -- dev --
from dev_basics.utils.metrics import compute_psnrs,compute_ssims

# -- superpixel --
from spix_paper import data
from spix_paper import utils
from spix_paper.attn import SuperpixelAttention
from spix_paper.attn import NeighAttnMat,NeighAttnAgg
from spix_paper.connected import connected_sp
from spix_paper.spix_utils import run_slic
from spix_paper.attn import run_search_fxn,run_stnls_search
from spix_paper.attn import run_stnls_agg,run_nat_agg

# -- nicer viz --
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def init_sna(attn_type,dist_type,normz,ksize,scale=20.,nftrs=3):

    # -- init --
    model = SuperpixelAttention(nftrs,normz_patch=normz,
                                kernel_size=ksize,attn_type=attn_type,
                                dist_type=dist_type,qk_scale=scale,
                                proj_layer=False)
    # -- id weights --
    with th.no_grad():
        model.nat_attn.qk.weight[:3,:] = th.eye(nftrs)
        model.nat_attn.qk.weight[-3:,:] = th.eye(nftrs)
        model.nat_agg.v.weight[...] = th.eye(nftrs)

    # -- add hooks --
    return model.eval()

# def viz_attn(root,name,attn,hi,wi,ksize):
#     # print(attn.shape)
#     amap = rearrange(attn[0,0,hi,wi],'(k1 k2) -> k1 k2',k1=ksize)
#     amap = amap - amap.min()
#     amap = amap/amap.max()
#     amap = interp(amap[None,:],256,256)
#     save_image(amap,root / ("%s.jpg"%name))
#     return amap.repeat(3,1,1).cpu()

def viz_square(img,hmid,wmid,ksize): # ,vsize

    # -- init --
    img = img[0].cpu()

    # -- viz box --
    sh,sw = hmid-ksize//2,wmid-ksize//2
    eh,ew = sh+ksize,sw+ksize
    # print("sh,sw: ",sh,sw,hmid,wmid,eh,ew,ksize)
    # exit()
    kbox = th.tensor([sw,sh,ew,eh])
    img = viz_loc_box(img,kbox)

    # -- get smaller crop within image --
    # sh,sw = hmid-vsize//2,wmid-vsize//2
    # eh,ew = sh+vsize,sw+vsize
    # img = img[:,sh:eh,sw:ew]
    # hmid = img_c.shape[0]//2
    # wmid = img_c.shape[1]//2
    # img = crop(img,hmid,wmid,vsize)

    return img[None,:]

def crop(img,hmid,wmid,vsize):
    # -- get smaller crop within image --
    sh,sw = hmid-vsize//2,wmid-vsize//2
    eh,ew = sh+vsize,sw+vsize
    img = img[...,sh:eh,sw:ew]
    return img

def divBy2(img):
    H,W = img.shape[-2:]
    return interp(img,H//2,W//2)

# def interp(img,tH,tW):
#     img = TF.resize(img,(tH,tW),interpolation=InterpolationMode.NEAREST)
#     return img

def get_image(dataset,ix,sigma,device):
    img = dataset[ix][0].to(device)[None,:]
    noisy = img + sigma*th.randn_like(img)
    return img/255.,noisy/255.

def overlay_spix(img,spix):
    if th.is_tensor(spix): spix = spix.cpu().numpy().astype(np.int64)
    cmin,cmax = 0.1,2
    # spix = connected_sp(spix,cmin,cmax)
    _img = img.cpu().numpy().transpose(1,2,0)
    # print(_img.shape,spix.shape)
    viz = mark_boundaries(_img,spix,mode="subpixel")
    viz = th.from_numpy(viz.transpose(2,0,1))
    return viz

def view_superpixels(img,spix,root,name):
    viz = overlay_spix(img,spix)
    # if th.is_tensor(spix): spix = spix.cpu().numpy().astype(np.int64)
    # cmin,cmax = 0.1,2
    # spix = connected_sp(spix,cmin,cmax)
    # _img = img.cpu().numpy().transpose(1,2,0)
    # # print(_img.shape,spix.shape)
    # viz = mark_boundaries(_img,spix,mode="subpixel")
    # viz = th.from_numpy(viz.transpose(2,0,1))
    save_image(viz,root / ("%s.jpg"%name))

# def viz_attn(root,name,attn,hi,wi,ksize):
#     # print("attn.shape: ",attn.shape)
#     eps = 1e-15
#     amap = rearrange(attn[0,0,hi,wi],'(k1 k2) -> k1 k2',k1=ksize)
#     # amap = amap - amap.min()
#     # amap = amap/(eps+amap.max())
#     amap = interp(amap[None,:],256,256)
#     save_image(amap,root / ("%s.jpg"%name))
#     return amap.repeat(3,1,1).cpu()

def index_attn(attn,hi,wi,ksize):
    # print(attn.shape)
    # exit()
    amap = rearrange(attn[:,0,hi,wi],'b (k1 k2) -> b k1 k2',k1=ksize)
    return amap

def viz_loc_box(img,box,col="red"):
    # img = rearrange(img,'h w c -> c h w')
    # if img.max().item() < 2:
    img = th.clamp(255.*img,0.,255.).type(th.uint8)
    # img = th.from_numpy(img).type(th.uint8)
    # box = th.tensor(box)
    img = draw_bounding_boxes(img,box[None,:],fill=True,colors=col)/255.
    # img = rearrange(img,'c h w -> h w c')
    return img


def interp(img,tH,tW):
    img = TF.resize(img,(tH,tW),interpolation=InterpolationMode.NEAREST)
    return img


def mark_spix(img,spix,hs,ws):
    B,F,H,W = img.shape
    for bi in range(B):
        args = th.where(spix[bi,hs,ws] == spix[bi])
        for fi in range(F):
            img[bi,fi][args] = fi==2
    return img
    # print(img.shape)
    # print(spix.shape)
    # print("mark.")
    # exit()
    # pass

def normalize(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor

def attn_hooks(attn_map,attn_key,model):
    def forward_hook(module, input, output):
        attn_map[attn_key] = input[1] # input = (x,attn)
    hook = model.nat_agg.register_forward_hook(forward_hook)
    return hook

def get_viz_grid(models,attn,clean,noisy,sims,spix,ksize,hs,ws,num,root):
    # print("hs,ws,ksize: ",hs,ws,ksize)

    # -- run models --
    B = len(clean)
    denos = {}
    for key,model in models.items():
        with th.no_grad():
            denos[key] = model(noisy,sims)

    # -- psnrs --
    psnrs = {}
    for atype in denos:
        psnrs[atype] = compute_psnrs(denos[atype],clean)
        psnrs[atype] = psnrs[atype].mean().item()
        print("[%s] psnr: %2.3f"%(atype,psnrs[atype]))
        save_image(denos[atype],root / ("deno_%s.jpg"%atype))
        denos[atype] = th.clip(denos[atype],0.,1.)

    # -- compat --
    deno0 = denos['na'].clone().cpu()
    deno1 = denos['soft'].clone().cpu()
    deno2 = denos['hard'].clone().cpu()
    out0 = denos['na'].clone()
    out1 = denos['soft'].clone()
    out2 = denos['hard'].clone()
    attn0 = attn['na']
    attn1 = attn['soft']
    attn2 = attn['hard']
    g_psnr0 = psnrs['na']
    g_psnr1 = psnrs['soft']
    g_psnr2 = psnrs['hard']
    # marked = clean.cpu().clone()

    # -- color superpixel in deno --
    # for atype in denos:
    #     for bi in range(B):
    #         for i in range(3):
    #             args = th.where(spix[bi] == spix[bi,hs,ws])
    #             denos[atype][bi,i][args] = i==2

    # -- format --
    out0,out1,out2 = out0.cpu(),out1.cpu(),out2.cpu()
    attn0,attn1,attn2 = attn0.cpu(),attn1.cpu(),attn2.cpu()
    hgrid = th.arange(hs-num//2,hs-num//2+num)
    wgrid = th.arange(ws-num//2,ws-num//2+num)
    grid = th.cartesian_prod(hgrid,wgrid)
    viz0 = th.zeros((B,len(hgrid)+ksize,len(wgrid)+ksize))
    hstart = hs-num//2-ksize//2
    hend = hstart + num + ksize
    wstart = ws-num//2-ksize//2
    wend = wstart + num + ksize
    # print(hstart,hend,(hend+hstart)/2.,hs)
    # exit()
    out0 = out0[...,hstart:hend,wstart:wend]
    out1 = out1[...,hstart:hend,wstart:wend]
    out2 = out2[...,hstart:hend,wstart:wend]
    clean_crop = clean[...,hstart:hend,wstart:wend].cpu()
    viz1,viz2 = viz0.clone(),viz0.clone()
    nrmz = viz0.clone()
    for (hi,wi) in grid:
        ix = hi - th.min(hgrid)
        jx = wi - th.min(wgrid)
        _attn0 = index_attn(attn0,hi,wi,ksize)
        _attn1 = index_attn(attn1,hi,wi,ksize)
        _attn2 = index_attn(attn2,hi,wi,ksize)
        # print(th.mean(_attn0.abs()),th.mean(_attn1.abs()),th.mean(_attn2.abs()))
        for bi in range(B):
            if spix[0,hs,ws] != spix[0,hi,wi]: continue
            viz0[bi,ix:ix+ksize,jx:jx+ksize] += _attn0[bi]
            viz1[bi,ix:ix+ksize,jx:jx+ksize] += _attn1[bi]
            viz2[bi,ix:ix+ksize,jx:jx+ksize] += _attn2[bi]
            nrmz[bi,ix:ix+ksize,jx:jx+ksize] += 1.
            # # -- mark spix --
            # marked = interp(viz_square(mark_spix(marked,spix,hs,ws),
            #                            hs,ws,psize),tH,tW)[0]


    # -- psnrs --
    psnr0 = compute_psnrs(out0,clean_crop)[0]
    psnr1 = compute_psnrs(out1,clean_crop)[0]
    psnr2 = compute_psnrs(out2,clean_crop)[0]

    # -- mask outside of spix --
    for i in range(3):
        out0[:,i][th.where(nrmz==0)] = 0.
        out1[:,i][th.where(nrmz==0)] = 0.
        out2[:,i][th.where(nrmz==0)] = 0.
        clean_crop[:,i][th.where(nrmz==0)] = 0.

    # -- view psnrs --
    delta0 = th.mean((out0-clean_crop)**2,1)[th.where(nrmz>0)]
    delta1 = th.mean((out1-clean_crop)**2,1)[th.where(nrmz>0)]
    delta2 = th.mean((out2-clean_crop)**2,1)[th.where(nrmz>0)]
    _psnr0 = -10*th.log10(th.mean(delta0)).item()
    _psnr1 = -10*th.log10(th.mean(delta1)).item()
    _psnr2 = -10*th.log10(th.mean(delta2)).item()
    print("[na]  PSNR: %2.3f, %2.3f, %2.3f" % (g_psnr0,psnr0,_psnr0))
    print("[sna] PSNR: %2.3f, %2.3f, %2.3f "% (g_psnr1,psnr1,_psnr1))
    print("[hna] PSNR: %2.3f, %2.3f, %2.3f "% (g_psnr2,psnr2,_psnr2))

    # -- normalize --
    eps = 1e-15
    viz0 = viz0 / (eps + nrmz)
    viz0 = viz0 - viz0.min()
    viz0 = viz0 / th.quantile(viz0,th.tensor([0.99])).item()

    # -- normalize --
    viz1 = viz1 / (eps + nrmz)
    viz1 = viz1 - viz1.min()
    viz1 = viz1 / th.quantile(viz1,th.tensor([0.99])).item()

    # -- normalize --
    viz2 = viz2 / (eps + nrmz)
    viz2 = viz2 - viz2.min()
    viz2 = viz2 / th.quantile(viz2,th.tensor([0.99])).item()

    # -- cat with nrmz as alpha channel --
    alpha = 1.*(nrmz[:,None]>0)
    # print(alpha.shape,viz0.shape)
    viz0 = th.cat([viz0[:,None].repeat(1,3,1,1),alpha],1)
    viz1 = th.cat([viz1[:,None].repeat(1,3,1,1),alpha],1)
    viz2 = th.cat([viz2[:,None].repeat(1,3,1,1),alpha],1)
    out0 = th.cat([out0,alpha],1)
    out1 = th.cat([out1,alpha],1)
    out2 = th.cat([out2,alpha],1)
    # deno0 = th.cat([deno0,alpha],1)
    # deno1 = th.cat([deno1,alpha],1)
    # deno2 = th.cat([deno2,alpha],1)


    # -- clean cropped region --
    clean_crop = th.cat([clean_crop,alpha],1)
    clean_crop = interp(clean_crop,256,256)
    # clean_crop = th.cat([clean_crop,th.ones_like(clean_crop[:,:1])],1)

    # -- interpolation --
    msize = 32
    vsize = 128
    viz0 = interp(viz0,256,256)#.repeat(1,3,1,1)
    viz1 = interp(viz1,256,256)#.repeat(1,3,1,1)
    viz2 = interp(viz2,256,256)#.repeat(1,3,1,1)
    out0 = interp(out0,256,256)
    out1 = interp(out1,256,256)
    out2 = interp(out2,256,256)
    # clean = interp(viz_square(mark_spix(clean.cpu(),spix,hs,ws),
    #                           hs,ws,num),256,256)

    # clean = interp(mark_spix(clean.cpu(),spix,hs,ws),256,256)
    # deno0 = interp(mark_spix(deno0.cpu(),spix,hs,ws),256,256)
    # deno1 = interp(mark_spix(deno1.cpu(),spix,hs,ws),256,256)
    # deno2 = interp(mark_spix(deno2.cpu(),spix,hs,ws),256,256)

    _clean = overlay_spix(clean[0].cpu(),spix[0])[None,:]
    # _deno0 = overlay_spix(deno0[0].cpu(),spix[0])[None,:]
    # _deno1 = overlay_spix(deno1[0].cpu(),spix[0])[None,:]
    # _deno2 = overlay_spix(deno2[0].cpu(),spix[0])[None,:]

    _clean = crop(_clean,2*hs,2*ws,2*num)
    # _deno0 = crop(_deno0,2*hs,2*ws,2*num)
    # _deno1 = crop(_deno1,2*hs,2*ws,2*num)
    # _deno2 = crop(_deno2,2*hs,2*ws,2*num)

    pad = th.nn.functional.pad
    _clean = th.cat([_clean,th.ones_like(_clean[:,:1])],1)
    clean = interp(pad(_clean,(5,5,5,5)),256,256)
    # print(clean.shape,viz0.shape,clean_crop.shape,out0.shape)
    # deno0 = interp(_deno0,256,256)
    # deno1 = interp(_deno1,256,256)
    # deno2 = interp(_deno2,256,256)

    # print(viz0.shape,viz1.shape,out0.shape,out1.shape,zero.shape)
    # print(clean_crop.shape,out0.shape)
    zero = th.zeros_like(viz0)
    grid = th.stack([clean,viz0,viz1,viz2,
                     clean_crop,out0,out1,out2])
    # print(grid.shape)
    grid = grid[:,0]
    # print(grid.shape)
    # exit()
    # print(viz0.min(),viz0.max(),viz1.min(),viz1.max(),
    #       out0.min(),out0.max(),out1.min(),out1.max())
    # print(grid.shape)


    return grid,denos,psnrs

def main():

    # -- config --
    root = Path("output/attn_rw/")
    if not root.exists(): root.mkdir()
    ix = 1
    sigma = 25.
    sigma_sp = 10.
    device = "cuda"

    # -- read and unpack configs --
    # seed = 123
    seed = 234
    utils.seed_everything(seed)
    device = "cuda:0"
    ksize = 15
    # psize = 96

    # -- superpixel config --
    sp_stride = 14
    sp_niters = 10
    sp_scale = 30.
    sp_m = sp_scale*0.75
    # sp_scale = 20.
    grad_type = "full"

    # -- superpixel attn --
    nftrs = 3
    nheads = 1
    qk_scale = 5.
    dilation = 1
    normz = True
    dist_type = "l2"

    # -- optimal "qk_val" --
    _sigma = sigma/255.
    print("1/(2.*sigma**2): ",1/(2*_sigma*_sigma))

    # -- load data --
    # vid_names = ["tuk-tuk"]
    # vid_names = ["kite-walk"]
    vid_names = ["tennis"]
    clean = data.davis_example(isize=None,nframes=10,vid_names=vid_names)
    clean = divBy2(clean[0,:1,:,:480,:480])
    # clean = clean[0,:3,:,:480,-480:]
    # clean = clean[0,:3,:,:256,:256]
    noise = th.randn_like(clean)
    noisy = clean + (sigma/255.) * noise
    noisy_sp = clean + (sigma_sp/255.) * noise
    B,F,H,W = clean.shape
    tv_utils.save_image(clean,root/"vid.png")

    # -- compute superpixels --
    print("clean.shape: ",clean.shape)
    # sims = run_slic(clean, sp_stride, sp_niters, sp_m, sp_scale, grad_type)[1]
    # sims = run_slic(noisy, sp_stride, sp_niters, sp_m, sp_scale, grad_type)[1]
    sims = run_slic(noisy_sp, sp_stride, sp_niters, sp_m, sp_scale, grad_type)[1]
    spix = sims.argmax(1).reshape(B,H,W)
    shape_str = 'b (sh sw) (h w) -> b h w sh sw'
    sims = rearrange(sims,shape_str,h=H,sh=H//sp_stride)

    # -- view superpixels --
    view_superpixels(clean[0],spix[0],root,"spix")

    # -- check identity --
    # deno_na = model_na(noisy)
    # deno_sna = model_sna(noisy)
    # deno_hna = model_hna(noisy)
    # # -- load dataset --
    # device = "cuda:0"
    # sigma = cfgs.na.sigma

    # -- init info --
    psize = 48
    # hs,ws = 145,165 # leg
    # hs,ws = 80,102 # text
    hs,ws = 40,102 # text

    # clean[...,140:160,140:160] = 0.
    # tv_utils.save_image(clean,root/"vid.png")
    # exit()

    # -- compute grid --
    # qk_scale_grid = np.linspace(1,20,15).tolist()
    qk_scale_grid = np.linspace(10,30,5).tolist()
    qk_scale_grid = [0.1,] + qk_scale_grid#1.0,2.0,3.,5.0,7.5,10.0]
    qk_str_grid = [("%2.1f"%v).replace(".","p") for v in qk_scale_grid]
    # print(qk_str_grid)
    # exit()
    for qk_scale,qk_str in zip(qk_scale_grid,qk_str_grid):
        print("QK Scale: ",qk_scale)

        # -- init sna --
        models,hooks = {},{}
        attn_types = ["na","soft","hard"]
        attn_maps = {k:[] for k in attn_types}
        for atype in attn_types:
            models[atype]=init_sna(atype,dist_type,normz,ksize,scale=qk_scale).to(device)
            hooks[atype]=attn_hooks(attn_maps,atype,models[atype])

        # -- load grid --
        # print("spix.shape: ",spix.shape)
        grid,deno,psnrs = get_viz_grid(models,attn_maps,clean,noisy,
                                       sims,spix,ksize,hs,ws,psize,root)

        # -- create grid --
        # print(grid[0].shape)
        grid = make_grid(grid,nrow=4)
        # print(grid.shape)
        grid = grid[...,2:-2,:]
        tH = grid.shape[-2]
        tW = int(tH/(1.*H)*W)
        # print("tH,tW: ",tH,tW)
        _clean = interp(viz_square(mark_spix(clean.cpu(),spix,hs,ws),
                                   hs,ws,psize),tH,tW)[0]
        _clean = th.cat([_clean,th.ones_like(_clean[:1])],0)
        # print(grid.shape,_clean.shape)
        # _clean = th.nn.functional.pad(_clean,(1,1,1,1))
        # print(grid.shape,_clean.shape)
        grid = th.cat([_clean,grid],-1)
        # grid = make_grid([_clean[0],grid],nrow=1)
        # grid = th.cat([_clean[0],grid],-2)
        # print(grid.shape,_clean.shape)
        # exit()
        fname = root/("grid_%s.png"%qk_str)
        print("fname: ",fname)
        save_image(grid,fname,nrow=4)

        # -- save deno --
        # _deno = th.cat([deno[atype] for atype in attn_types])
        # grid = th.cat([_deno,clean])
        # save_image(grid,root/("deno_%s.png"%qk_str),nrow=B)

        # -- remove hooks --
        for name,hook in hooks.items():
            hook.remove()


if __name__ == "__main__":
    main()
