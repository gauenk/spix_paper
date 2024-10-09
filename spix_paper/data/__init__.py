
# -- datasets --
from .bsd500_seg import load_bsd500
# from .dnd import load_dnd
from .simple import davis_example
from ..utils import optional
from .benchmark import load_eval_set

def load_data(cfg):
    dname = optional(cfg,"dname","noname")
    load_test = optional(cfg,"data_load_test",False)
    if dname == "bsd500":
        return load_bsd500(cfg,load_test=load_test)
    elif dname in ["set5","manga109","urban100","b100","bsd100","bsd68"]:
        assert load_test == True,"Must be test time."
        return load_eval_set(cfg.data_path,dname)
    else:
        raise ValueError(f"Uknown model type [{dname}]")

