
# -- basic --
import torch
import torch as th
import torch.nn as nn

# -- losses --
from .sp_loss import SuperpixelLoss
from .interface import LossInterface
from ..utils import optional

def load_loss(cfg):
    lname = optional(cfg,"lname","spix")
    target = optional(cfg,"spix_loss_target","seg")
    alpha = optional(cfg,"deno_spix_alpha",0.5)
    sp_loss = None
    if "spix" in lname:
        spix_loss_type = optional(cfg,"spix_loss_type","cross")
        spix_loss_compat = optional(cfg,"spix_loss_compat",0.)
        sp_loss = SuperpixelLoss(spix_loss_type,spix_loss_compat)
    loss = LossInterface(sp_loss,lname,target,alpha)
    return loss
