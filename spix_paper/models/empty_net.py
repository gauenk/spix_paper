
import torch as th
import torch.nn as nn
from spix_paper.spix_utils.slic_img_iter import run_slic

class EmptyNet(nn.Module):

    def __init__(self, slic_iters=5, slic_stride=8, slic_scale=1,
                 slic_M=0.1, slic_grad=True):
        super(EmptyNet, self).__init__()
        self.slic_iters = slic_iters
        self.slic_stride = slic_stride
        self.slic_scale = slic_scale
        self.slic_M = slic_M
        self.slic_grad = slic_grad

    def get_stride(self):
        if not hasattr(self.slic_stride,"__len__"):
            stride = [self.slic_stride,self.slic_stride]
        else:
            stride = self.slic_stride
        if len(stride) == 1:
            stride = [self.slic_stride[0],self.slic_stride[0]]
        return stride

    def forward(self, x):
        stride = self.get_stride()
        _slic = run_slic(x, M = self.slic_M,
                         stoken_size=stride,n_iter=self.slic_iters,
                         sm_scale=self.slic_scale,grad_type=self.slic_grad)
        sims = _slic[1]
        return {"sims":sims}

    def get_superpixel_probs(self, x):
        return self(x)

