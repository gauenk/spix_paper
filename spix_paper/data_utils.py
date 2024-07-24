
import random
import torch as th
from .utils import ndarray2tensor

def crop_patch(lr, hr, patch_size, augment=True):
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape
    hp = patch_size
    lp = patch_size
    lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
    hx, hy = lx, ly
    lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp]
    # augment data
    # print("[top]: ",lr_patch.shape,hr_patch.shape)
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip:
            lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1]
        if vflip:
            lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :]
        if rot90:
            lr_patch, hr_patch = lr_patch.transpose(1,0,2), hr_patch.transpose(1,0)
        # numpy to tensor
    # print("[in]: ",lr_patch.shape,hr_patch.shape)
    lr_patch = ndarray2tensor(lr_patch).contiguous()
    hr_patch = th.from_numpy(1.*hr_patch.copy()).float()
    return lr_patch, hr_patch

