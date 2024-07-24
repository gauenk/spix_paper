
"""

Training

"""


# -- sys --
import os
import numpy as np
import pandas as pd

# -- training --
from spix_paper.trte_spix import test

# -- caching results --
import cache_io

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)
    read_testing = False

    # -- get/run experiments --
    refresh = True and not(read_testing)
    def clear_fxn(num,cfg): return False
    read_test = cache_io.read_test_config.run

    # -- load experiments --
    train_fn_list = [
        "exps/comp_graphs/train.cfg",
        "exps/comp_graphs/train_empty.cfg",
    ]
    te_fn = "exps/comp_graphs/test_shell.cfg"
    exps,uuids = [],[]
    for tr_fn in train_fn_list:
        is_empty = "empty" in tr_fn # special load; no network
        tr_exp = cache_io.fill_test_shell(tr_fn,te_fn)
        _exps = read_test(tr_exp,".cache_io_exps/comp_graphs/test",
                          reset=refresh,skip_dne=False,keep_dne=is_empty)
        _exps,_uuids = cache_io.get_uuids(_exps,".cache_io/comp_graphs/test",
                                          read=not(refresh),no_config_check=False)
        exps += _exps
        uuids += _uuids


    # -- run exps --
    results = cache_io.run_exps(exps,test.run,uuids=uuids,preset_uuids=True,
                                name=".cache_io/comp_graphs/test",
                                version="v1",skip_loop=False,clear_fxn=clear_fxn,
                                clear=False,enable_dispatch="slurm",
                                records_fn=".cache_io_pkl/comp_graphs/test.pkl",
                                records_reload=True and not(read_testing),
                                use_wandb=False,proj_name="comp_graphs_test")

    # -- view results --
    results = results.fillna(value=-1)
    print(results)
    vfields0 = ["asa","br","bp"]
    vfields1 = ["psnr","ssim","entropy"]
    vfields = vfields0 + vfields1
    gfields = ["slic_grad"]
    results0 = results.groupby(gfields, as_index=False).agg(
        {k:['mean','std'] for k in vfields0})
    print(results0)
    results1 = results.groupby(gfields, as_index=False).agg(
        {k:['mean','std'] for k in vfields1})
    print(results1)



if __name__ == "__main__":
    main()

