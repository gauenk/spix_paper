

# -- models --
from . import unet_ssn
from . import empty_net
from .simple_deno import SimpleDenoiser
from .attn_scale_net import AttnScaleNet
from ..utils import optional,get_fxn_kwargs,extract_defaults_new

def load_model(cfg):
    mname = optional(cfg,"mname","default")
    if mname == "unet_ssn":
        fxn = unet_ssn.UNetSsnNet.__init__
        kwargs = get_fxn_kwargs(cfg,fxn)
        model = unet_ssn.UNetSsnNet(**kwargs)
    elif mname == "empty":
        fxn = empty_net.EmptyNet.__init__
        kwargs = get_fxn_kwargs(cfg,fxn)
        model = empty_net.EmptyNet(**kwargs)
    elif mname == "simple_deno":
        fxn = SimpleDenoiser.defs
        kwargs = extract_defaults_new(cfg,fxn)
        print(dict(kwargs))
        model = SimpleDenoiser(cfg.dim,**kwargs)
    else:
        raise ValueError(f"Uknown model type [{mname}]")
    return model

