

import torch as th
import numpy as np
from skimage.segmentation.slic_superpixels import _enforce_label_connectivity_cython

def connected_sp(_spix,min_size_factor=0.3, max_size_factor=3):
    if _spix.ndim == 2:
        return _connected_sp(_spix,min_size_factor,max_size_factor)

    spix = []
    for bi in range(_spix.shape[0]):
        spix.append(_connected_sp(_spix[bi],min_size_factor,max_size_factor))
    spix = np.stack(spix)
    return spix

def _connected_sp(spix,min_size_factor=0.3, max_size_factor=3):

    if th.is_tensor(spix):
        spix = spix.cpu().numpy()

    H,W = spix.shape
    spix = spix.astype(np.int)
    start_label = 0
    Nsp = int(spix.max())+1

    segment_size = H*W/Nsp
    min_size = int(min_size_factor * segment_size)
    max_size = int(max_size_factor * segment_size)

    labels = _enforce_label_connectivity_cython(
        spix[None,:], min_size, max_size, start_label=start_label
    )
    labels = labels[0]

    return labels
