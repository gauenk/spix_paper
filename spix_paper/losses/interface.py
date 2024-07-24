# -- basic --
import torch
import torch as th
import torch.nn as nn

class LossInterface(nn.Module):

    def __init__(self,sp_loss,lname,spix_target,mix_alpha):
        super().__init__()
        self.sp_loss = sp_loss
        self.lname = lname
        self.spix_target = spix_target
        self.mix_alpha = mix_alpha

    def spix_loss(self,img,seg,sims):
        if self.spix_target == "seg":
            return self.sp_loss(seg,sims)
        elif self.spix_target == "pix":
            return self.sp_loss(img,sims)
        else:
            raise ValueError(f"Uknown target [{self.spix_loss_target}]")

    def forward(self, img, seg, deno=None, sims=None):
        if self.lname == "spix":
            return self.spix_loss(img,seg,sims)
        elif self.lname == "deno":
            return th.sqrt(th.mean((img-deno)**2)+1e-6) # deno loss
        elif self.lname == "deno+spix":
            loss0 = th.sqrt(th.mean((img-deno)**2)+1e-6) # deno loss
            loss1 = self.spix_loss(img,seg,sims)
            loss = self.mix_alpha * loss0 + (1-self.mix_alpha) * loss1
            return loss
        else:
            raise ValueError(f"Uknown lname [{self.lname}]")
