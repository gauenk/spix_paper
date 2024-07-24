
"""

Training

"""


# -- sys --
import os
import copy
dcopy = copy.deepcopy
import torch as th
import numpy as np
import pandas as pd

# -- testing --
from st_spix.data import load_data
from st_spix.models import load_model

# -- caching results --
import cache_io

# -- dev --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt

def run_exp(exp):

    # -- read data --
    device = "cuda:0"
    model = load_model(exp).to("cuda")
    train_dataloader, valid_dataloaders = load_data(exp)

    # -- init cuda --
    img,seg = next(iter(train_dataloader))
    img = img.to(device)/255.

    # -- run once --
    sims = model(img)
    loss = sims.mean()
    loss.backward()
    model.zero_grad()
    th.cuda.synchronize()

    # -- time/mem benchmark --
    timer = ExpTimer()
    memer = GpuMemer()

    # -- forward --
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            out = model(img)

    # -- output --
    loss = out.mean()
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss.backward()

    # -- results --
    info = {}
    for name,(mem_res,mem_alloc) in memer.items():
        info["%s_res"%name] = mem_res
        info["%s_alloc"%name] = mem_alloc
    for name,time in timer.items():
        info[name] = time

    return info

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)
    version = "v1"
    device = "cuda:0"
    def clear_fxn(num,cfg): return False

    # -- get experiments --
    train_fn_list = [
        "exps/comp_graphs/bench.cfg",
    ]
    exps = []
    cache_fn = ".cache_io_exps/comp_graphs/bench/"
    for exp_fn in train_fn_list:
        _exps,_ = cache_io.train_stages.run(exp_fn,cache_fn,
                                            fast=False,update=True,
                                            cache_version=version)
        exps += _exps
    print("[original] Num Exps: ",len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,run_exp,
                                name=".cache_io/comp_graphs/bench",
                                version=version,skip_loop=False,
                                clear_fxn=clear_fxn,clear=False,
                                enable_dispatch="slurm",
                                records_fn=
                                ".cache_io_pkl/comp_graphs/bench.pkl",
                                records_reload=True,use_wandb=False,
                                proj_name="comp_graphs_bench")
    # -- info --
    print(results)
    print(results.columns)
    results = results.rename(columns={'unet_features':"nf",
                                      "slic_stride":"st","slic_iters":"ni"})
    gnames = ["slic_grad","nf","st","ni"]
    vnames0 = ["fwd_alloc","bwd_alloc"]
    vnames1 = ["timer_fwd","timer_bwd"]
    results = results[gnames+vnames0+vnames1]
    results0 = results.groupby(gnames, as_index=False).agg(
        {k:['mean','std'] for k in vnames0})
    results1 = results.groupby(gnames, as_index=False).agg(
        {k:['mean','std'] for k in vnames1})
    with pd.option_context('display.float_format', '{:0.4f}'.format):
        print(results0.sort_values(gnames[1:]))
        print(results1.sort_values(gnames[1:]))

    # for group,gdf in results[names].groupby("slic_grad"):
    #     print(group)
    #     print(gdf[names[:-1]])
    #     print(gdf[names[:-1]].mean())
    #     print(gdf[names[:-1]].std())

if __name__ == "__main__":
    main()
