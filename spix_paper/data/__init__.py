
# -- datasets --
from .bsd500_seg import load_bsd500
from .dnd import load_dnd
from .simple import davis_example
from ..utils import optional

def load_data(cfg):
    dname = optional(cfg,"dname","default")
    if dname == "bsd500":
        load_test = optional(cfg,"data_load_test",False)
        return load_bsd500(cfg,load_test=load_test)
    elif dname == "dnd":
        return load_dnd(cfg)
    else:
        raise ValueError(f"Uknown model type [{dname}]")

