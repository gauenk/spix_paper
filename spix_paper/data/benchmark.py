import os
import glob
import random
import pickle
from pathlib import Path

import numpy as np
import imageio
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.utils.data as data
import skimage.color as sc
from torch.utils.data import DataLoader
import time
from ..utils import ndarray2tensor

def load_eval_set(data_path,dname):
    hr_path = os.path.join(data_path, f'{dname}/HR')
    lr_path = os.path.join(data_path, f'{dname}/LR_bicubic')
    dset = Benchmark(hr_path, lr_path, scale=1, colors=3)
    dataloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    return dset,dataloader

class Benchmark(data.Dataset):
    def __init__(self, HR_folder, LR_folder, scale=2, colors=1):
        super(Benchmark, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder

        self.img_postfix = '.png'
        self.in_scale = scale
        scale = 2 if scale == 1 else scale
        self.scale = scale
        self.colors = colors

        self.nums_dataset = 0

        self.hr_filenames = []
        self.lr_filenames = []
        ## generate dataset
        tags = sorted(os.listdir(self.HR_folder))
        for tag in tags:
            hr_filename = os.path.join(self.HR_folder, tag)
            lr_filename = os.path.join(self.LR_folder, 'X{}'.format(scale), tag.replace('.png', 'x{}.png'.format(self.scale)))
            if not Path(lr_filename).exists():
                lr_filename = os.path.join(self.LR_folder, 'X{}'.format(scale),tag)
                assert Path(lr_filename).exists(),"lr_filename: %s" % lr_filename
            self.hr_filenames.append(hr_filename)
            self.lr_filenames.append(lr_filename)
        self.nums_trainset = len(self.hr_filenames)
        ## if store in ram
        self.hr_images = []
        self.lr_images = []
        self.names = []

        LEN = len(self.hr_filenames)
        for i in range(LEN):
            try:
                hr_image = imageio.imread(self.hr_filenames[i], pilmode="RGB")
                lr_image = imageio.imread(self.lr_filenames[i], pilmode="RGB")
            except:
                hr_image = np.array(Image.open(self.hr_filenames[i]))
                lr_image = np.array(Image.open(self.lr_filenames[i]))
            if self.colors == 1:
                lr_image, hr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1], sc.rgb2ycbcr(hr_image)[:, :, 0:1]
            self.hr_images.append(hr_image)
            self.lr_images.append(lr_image)
            self.names.append(Path(self.hr_filenames[i]).stem)

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, idx):
        # get whole image, store in ram by default
        lr, hr = self.lr_images[idx], self.hr_images[idx]
        lr_h, lr_w, _ = lr.shape
        hr = hr[:lr_h*self.scale, :lr_w*self.scale, :] # always "x2" lr, so a little smaller than normal hr
        lr, hr = ndarray2tensor(lr), ndarray2tensor(hr)
        if self.in_scale == 1: lr = hr.clone()
        seg = torch.zeros_like(hr[:0,:0,:0])
        # print("hr.shape: ",hr.shape,seg.shape)
        # exit()
        return hr, seg

if __name__ == '__main__':
    pass
