"""

   Inspect convolution filters.

"""

import os
from pathlib import Path

import torch as th

import cache_io

import spix_paper.trte_utils as utils
import spix_paper.utils as base_utils
from spix_paper.trte.train import tr_defs
from spix_paper.models import load_model as _load_model

import matplotlib.cm as cm
import matplotlib.pyplot as plt

def get_exps():
    # -- get experiments --
    version = "v1"
    def clear_fxn(num,cfg): return False
    exp_fn_list = [
        # "exps/trte_deno/train.cfg",
        # "exps/trte_deno/train_conv.cfg",
        "exps/trte_deno/train_lin.cfg",
        # "exps/trte_deno/train_conv_only.cfg",
        # "exps/trte_deno/train_empty.cfg",
        # "exps/trte_deno/train_spix.cfg",
    ]
    exps,uuids = [],[]
    cache_fn = ".cache_io_exps/trte_deno/train/"
    for exp_fn in exp_fn_list:
        _exps,_uuids = cache_io.train_stages.run(exp_fn,cache_fn,
                                                 fast=False,update=True,
                                                 cache_version=version)
        exps += _exps
        uuids += _uuids

    for e,u in zip(exps,uuids): e.uuid = u
    res = {e.attn_type: e for e,u in zip(exps,uuids)}
    return res

def load_model(cfg,device):

    # -- read module --
    cfg = base_utils.extract_defaults(cfg,tr_defs)
    cfg.tr_uuid = cfg.uuid
    model = _load_model(cfg)

    # -- restore from checkpoint --
    chkpt_root = utils.get_ckpt_root(cfg.tr_uuid,cfg.base_path)
    chkpt,chkpt_fn = utils.get_checkpoint(chkpt_root)
    loaded_epoch = utils.load_checkpoint(chkpt,model)
    print("Restoring State from [%s]" % chkpt_fn)
    model = model.to(device)
    model.eval()

    return model

def extract_kernels(model):

    weights = {}
    weights['conv0'] = model.conv0.weight.data.cpu()
    weights['conv1'] = model.conv1.weight.data.cpu()
    print(weights['conv1'].shape)
    for ix in range(len(model.attn)):
        data = model.attn[ix].proj_attn.weight.data
        weights['attn%d'%ix] = data.reshape(81,1,9,9).cpu()

    bias = {}
    bias['conv0'] = model.conv0.bias.data.cpu()
    bias['conv1'] = model.conv1.bias.data.cpu()
    print(bias['conv1'].shape)
    for ix in range(len(model.attn)):
        data = model.attn[ix].proj_attn.bias.data
        bias['attn%d'%ix] = data.reshape(1,1,9,9).cpu()
    # bias['attn0'] = model.attn[0].proj_attn.bias.data.reshape(1,1,9,9).cpu()

    return weights,bias

def view_params(root,name,atypes,params0,params1):
    if params0.ndim != 4: return
    print("Name: ",name)
    nrow,ncol,k0,k1 = params0.shape
    params0 = params0.reshape(nrow*ncol,k0,k1).abs()
    params1 = params1.reshape(nrow*ncol,k0,k1).abs()
    params0 = th.softmax(-params0.ravel(),0).reshape(params0.shape)
    params1 = th.softmax(-params1.ravel(),0).reshape(params1.shape)

    params = th.stack([params0,params1]).ravel()
    pmin = th.quantile(params,th.tensor([0.01])).item()
    pmin = th.min(params)
    params0 = params0 - pmin
    params1 = params1 - pmin

    params = th.stack([params0,params1]).ravel()
    pmax = th.quantile(params,th.tensor([0.99])).item()
    pmax = th.max(params)
    params0 = params0/pmax
    params1 = params1/pmax

    # print(params0.min(),params1.min(),params0.max(),params1.max(),
    #       params1.abs().mean(),pmax,pmin)
    # print(params0.abs().mean(),params1.abs().mean(),pmax,pmin)


    dpi = 200
    ginfo = {'wspace':0.01, 'hspace':0.01,
             "top":0.92,"bottom":0.16,"left":.07,"right":0.98}
    width = int(nrow*ncol)+2
    fig,axes = plt.subplots(2,nrow*ncol,figsize=(width,3),gridspec_kw=ginfo,dpi=200)
    if nrow*ncol == 1:
        axes[0] = [axes[0]]
        axes[1] = [axes[1]]
    # print(params0.shape,params1.shape)
    for r in range(nrow*ncol):
        # print(type(axes),type(axes[0]))
        # print(r,params0[r].min(),params0.max(),params1[r].min(),params1.max())
        axes[0][r].imshow(params0[r],vmin=0,vmax=1.,cmap=cm.gray)
        axes[1][r].imshow(params1[r],vmin=0,vmax=1.,cmap=cm.gray)
        axes[0][r].text(10, 5, atypes[0], bbox={'facecolor': 'white', 'pad': 10})
        axes[1][r].text(10, 5, atypes[1], bbox={'facecolor': 'white', 'pad': 10})

        axes[0][r].axis('off')
        axes[1][r].axis('off')
    plt.savefig(root / ("%s.png"%name))
    plt.close("all")
    # exit()

def main():
    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)
    device = "cuda"
    root = Path("./output/inspect_filters/")
    if not root.exists():
        root.mkdir(parents=True)

    # -- read exps --
    exps = get_exps()

    # -- load models --
    models,weights,bias = {},{},{}
    for atype,exp in exps.items():
        models[atype] = load_model(exp,device)
        weights[atype],bias[atype] = extract_kernels(models[atype])

    # -- view --
    atypes = list(weights.keys())
    wtypes = list(weights[atypes[0]].keys())
    # print(models)
    for wtype in wtypes:
        viz = [weights[atype][wtype] for atype in atypes]
        view_params(root,wtype+"_weights",atypes,*viz)
    for wtype in wtypes:
        viz = [bias[atype][wtype] for atype in atypes]
        view_params(root,wtype+"_bias",atypes,*viz)
    exit()


if __name__ == "__main__":
    main()
