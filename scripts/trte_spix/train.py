
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- training --
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
        "exps/trte_spix/train.cfg",
        "exps/trte_spix/train_empty.cfg",
    ]
    exps,uuids = [],[]
    cache_fn = ".cache_io_exps/trte_spix/train/"
    for exp_fn in exp_fn_list:
        _exps,_uuids = cache_io.train_stages.run(exp_fn,cache_fn,
                                                 fast=False,update=True,
                                                 cache_version=version)
        exps += _exps
        uuids += _uuids
    print("[original] Num Exps: ",len(exps))

    # -- run exps --
    results = cache_io.run_exps(exps,train.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/trte_spix/train",
                                version=version,skip_loop=False,
                                clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=
                                ".cache_io_pkl/trte_spix/train.pkl",
                                records_reload=False,use_wandb=False,
                                proj_name="trte_spix_train")



if __name__ == "__main__":
    main()
