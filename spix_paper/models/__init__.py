

# -- models --
from . import unet_ssn
from . import empty_net
from .simple_deno import SimpleDenoiser
from .conv_deno import ConvDenoiser
from .conv_only import ConvOnlyDenoiser
from .lin_deno import LinearDenoiser
from .attn_only import AttentionOnlyDenoiser
from .attn_scale_net import AttnScaleNet
from ..utils import optional,get_fxn_kwargs,extract_defaults_new

def load_model(cfg):
    mname = optional(cfg,"mname","default")
    in_dim = optional(cfg,"in_dim",3)
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
        model = SimpleDenoiser(in_dim,cfg.dim,**kwargs)
    elif mname == "lin_deno":
        fxn = LinearDenoiser.defs
        kwargs = extract_defaults_new(cfg,fxn)
        model = LinearDenoiser(in_dim,cfg.dim,**kwargs)
    elif mname == "conv_deno":
        fxn = ConvDenoiser.defs
        kwargs = extract_defaults_new(cfg,fxn)
        model = ConvDenoiser(in_dim,cfg.dim,**kwargs)
    elif mname == "conv_only":
        fxn = ConvOnlyDenoiser.defs
        kwargs = extract_defaults_new(cfg,fxn)
        model = ConvOnlyDenoiser(in_dim,cfg.dim,**kwargs)
    elif mname == "attn_only":
        fxn = AttentionOnlyDenoiser.defs
        kwargs = extract_defaults_new(cfg,fxn)
        model = AttentionOnlyDenoiser(in_dim,cfg.dim,**kwargs)
    else:
        raise ValueError(f"Uknown model type [{mname}]")
    return model

