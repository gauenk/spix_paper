
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- training --
# from spix_paper.trte_spix import train
from spix_paper.trte import train

# -- caching results --
import cache_io


def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)
    version = "v1"

    # -- get experiments --
    def clear_fxn(num,cfg): return False
    exp_fn_list = [
        "exps/trte_deno/train.cfg",
        # "exps/trte_deno/train_conv.cfg",
        # "exps/trte_deno/train_lin.cfg",
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
    print("[original] Num Exps: ",len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_deno/train",
                                version=version,skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/trte_deno/train.pkl",
                                records_reload=False,use_wandb=False,
                                proj_name="trte_deno_train")



if __name__ == "__main__":
    main()
