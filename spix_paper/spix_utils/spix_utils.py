import torch as th
from einops import rearrange
import torch.nn.functional as th_f
from torchvision.utils import draw_segmentation_masks

from skimage.segmentation import mark_boundaries

def viz_spix(img_batch,spix_batch,nsp):
    masks = []
    img_batch = (img_batch*255.).clip(0,255).type(th.uint8)
    for img, spix in zip(img_batch, spix_batch):
        # _masks = img
        # masks.append(img)

        # _masks = th_f.one_hot(spix,nsp).T.bool()
        # _masks = rearrange(_masks,'c (h w) -> c h w',h=img.shape[1])
        # masks.append(draw_segmentation_masks(img, masks=_masks, alpha=0.5))

        img = rearrange(img,'c h w -> h w c').cpu().numpy()/255.
        spix = rearrange(spix,'(h w) -> h w',h=img.shape[0]).cpu().numpy()
        print(img.shape,spix.shape)
        _masks = mark_boundaries(img,spix,mode="subpixel")
        _masks = th.from_numpy(_masks)
        _masks = rearrange(_masks,'h w c -> c h w')
        masks.append(_masks)

        # _masks = _masks[:3]
        # print("_masks.shape: ",_masks.shape)
        # masks.append(_masks)
    masks = th.stack(masks)#/255.
    # masks = th.stack(masks)/(1.*nsp)
    # masks = masks / masks.max()
    return masks

def pool_flow_and_shift_mean(flow,means,spix,ids,K):

    # -- get labels --
    K = means.shape[1]
    sims = th_f.one_hot(spix.long(),num_classes=K)*1.
    sims = rearrange(sims,'b h w nsp -> b nsp (h w)')

    # -- normalize across #sp for each pixel --
    sims_nmz = sims / (1e-15+sims.sum(-1,keepdim=True))# (B,NumSpix,NumPix) -> (B,NS,NP)
    sims = sims.transpose(-1,-2)

    # -- prepare flow --
    W = flow.shape[-1]
    flow = rearrange(flow,'b f h w -> b (h w) f')

    # -- compute "superpixel pooling" --
    flow_tmp = sims_nmz @ flow
    flow_sp = sims @ (flow_tmp)

    # -- pool means --
    print("tmp: ",flow_tmp.shape,means.shape,sims.shape,
          flow.shape,means.shape,len(th.unique(spix)))
    # exit()
    means[...,-2] = means[...,-2] + flow_tmp[...,0]
    means[...,-1] = means[...,-1] + flow_tmp[...,1]

    # print("ids.shape: ",ids.shape,means.shape)
    # print("ids.min(),ids.max(): ",ids.min(),ids.max())
    # _means = th.gather(means.clone(),1,ids).clone()
    # _means = th.gather(means.clone(),1,ids).clone()
    # print("Num previous superpixels: ",len(th.unique(spix_st[-1])))
    # print("Previous [Min,Max]: ",spix_st[-1].min().item(),spix_st[-1].max().item())

    # th.cuda.synchronize()
    # exit()

    # -- reshape --
    flow_sp = rearrange(flow_sp,'b (h w) f -> b f h w',w=W)

    return flow_sp,means


def sp_pool_from_spix(labels,spix):
    sims_hard = th_f.one_hot(spix.long())*1.
    sims_hard = rearrange(sims_hard,'b h w nsp -> b nsp (h w)')
    labels_sp = sp_pool(labels,sims_hard)
    return labels_sp

def sp_pool(labels,sims,re_expand=True):
    assert re_expand == True,"Only true for now."

    # -- normalize across #sp for each pixel --
    sims_nmz = sims / (1e-15+sims.sum(-1,keepdim=True))# (B,NumSpix,NumPix) -> (B,NS,NP)
    sims = sims.transpose(-1,-2)

    # -- prepare labels --
    W = labels.shape[-1]
    labels = rearrange(labels,'b f h w -> b (h w) f')

    # -- compute "superpixel pooling" --
    labels_sp = sims @ (sims_nmz @ labels)

    # -- reshape --
    labels_sp = rearrange(labels_sp,'b (h w) f -> b f h w',w=W)
    # print("a: ",labels.min(),labels.max())
    # print("b: ",labels_sp.min(),labels_sp.max())

    return labels_sp

def sp_pool_v0(img_batch,spix_batch,sims,S,nsp,method):
    pooled = []
    for img, spix in zip(img_batch, spix_batch):
        img = rearrange(img,'f h w -> h w f')
        pool = sp_pool_img(img,spix,sims,S,nsp,method)
        pooled.append(rearrange(pool,'h w f -> f h w'))
    pooled = th.stack(pooled)
    return pooled

def sp_pool_img_v0(img,spix,sims,S,nsp,method):
    H,W,F = img.shape
    if method in ["ssn","sna"]:
        sH,sW = (H+1)//S,(W+1)//S # add one for padding
    else:
        sH,sW = H//S,W//S # no padding needed

    is_tensor = th.is_tensor(img)
    if not th.is_tensor(img):
        img = th.from_numpy(img)
        spix = th.from_numpy(spix)

    img = img.reshape(-1,F)
    spix = spix.ravel()
    # N = nsp
    N = len(th.unique(spix))
    assert N <= (sH*sW)

    # -- normalization --
    counts = th.zeros((sH*sW),device=spix.device)
    ones = th.ones_like(img[:,0])
    counts = counts.scatter_add_(0,spix,ones)

    # -- pooled --
    pooled = th.zeros((sH*sW,F),device=spix.device)
    for fi in range(F):
        pooled[:,fi] = pooled[:,fi].scatter_add_(0,spix,img[:,fi])

    # -- exec normz --
    pooled = pooled/counts[:,None]

    # -- post proc --
    pooled = pooled.reshape(sH,sW,F)
    if not is_tensor:
        pooled = pooled.cpu().numpy()

    return pooled

