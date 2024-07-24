
# -- basic --
import os
import copy
dcopy = copy.deepcopy
import glob
import time
import random
import pickle
import imageio
from pathlib import Path
from easydict import EasyDict as edict

import skimage.color as sc
from scipy.io import loadmat

# -- torch --
import torch
import torch as th
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

# -- project --
import spix_paper.data_utils as utils
from ..utils import get_fxn_kwargs,optional

def load_bsd500(args,shuffle_train=True,load_test=False):

    # pairs = {"data_path":"./data","num_workers":4,"batch_size":1}
    data_path = optional(args,'data_path','./data')
    num_workers = optional(args,'num_workers',4)
    batch_size = optional(args,'batch_size',1)
    args = edict(get_fxn_kwargs(args,BSD500Seg.__init__)) # fill defaults

    dpath = Path(data_path)/'bsd500'
    cpath = dpath / ".cache"
    if load_test:
        dset = BSD500Seg(dpath,cpath,
                         "test",data_augment=args.data_augment,colors=args.colors,
                         patch_size=args.patch_size,data_repeat=args.data_repeat)
        test_dataloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
        return dset,test_dataloader


    dset = BSD500Seg(dpath,cpath,"train",
                     data_augment=args.data_augment,colors=args.colors,
                     patch_size=args.patch_size,data_repeat=args.data_repeat)
    train_dataloader = DataLoader(dataset=dset, num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train,
                                  pin_memory=True, drop_last=True)
    dset = BSD500Seg(dpath,cpath,
                     "val",data_augment=args.data_augment,colors=args.colors,
                     patch_size=args.patch_size,data_repeat=args.data_repeat)
    valid_dataloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)

    return train_dataloader,valid_dataloader

class BSD500Seg(data.Dataset):

    def __init__(
            self, ROOT_folder, CACHE_folder, split,
            data_augment=True, colors=3,
            patch_size=96, data_repeat=4, img_postfix=".png"):
        super(BSD500Seg, self).__init__()

        # -- set --
        train = split == "train"
        self.IMG_folder = Path(ROOT_folder)/("images/%s" % split)
        self.SEG_folder = Path(ROOT_folder)/("groundTruth/%s" % split)
        self.augment  = data_augment
        self.img_postfix = img_postfix
        self.colors = colors
        self.patch_size = patch_size
        self.data_repeat = data_repeat
        self.nums_trainset = 0
        self.train = train
        self.cache_dir = Path(CACHE_folder) / split

        ## for raw png images
        self.img_filenames = []
        self.seg_filenames = []
        ## for numpy array data
        self.img_npy_names = []
        self.seg_npy_names = []

        # ## store in ram
        # self.img_images = []
        # self.seg_images = []

        ## generate dataset
        self.start_idx = 0
        names = [p.stem for p in Path(self.IMG_folder).iterdir()]
        self.names = names
        self.end_idx = len(names)
        for i in range(self.start_idx, self.end_idx):
            idx = str(i).zfill(4)
            name = names[i]
            img_filename = os.path.join(self.IMG_folder, "%s.jpg" % name)
            seg_filename = os.path.join(self.SEG_folder, "%s.mat" % name)
            self.img_filenames.append(img_filename)
            self.seg_filenames.append(seg_filename)
        self.nums_trainset = len(self.img_filenames)
        LEN = self.end_idx - self.start_idx
        img_dir = os.path.join(self.cache_dir, 'bsd500_img',
                               'ycbcr' if self.colors==1 else 'rgb')
        seg_dir = os.path.join(self.cache_dir,"bsd500_seg",
                               'ycbcr' if self.colors==1 else 'rgb')

        # -- image --
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        else:
            for i in range(LEN):
                img_fn_i = self.img_filenames[i]
                img_npy_name = img_fn_i.split('/')[-1].replace('.jpg', '.npy')
                img_npy_name = os.path.join(img_dir, img_npy_name)
                self.img_npy_names.append(img_npy_name)

        # -- segmentation --
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        else:
            for i in range(LEN):
                seg_fn_i = self.seg_filenames[i]
                seg_npy_name = seg_fn_i.split('/')[-1].replace('.mat', '.npy')
                seg_npy_name = os.path.join(seg_dir, seg_npy_name)
                self.seg_npy_names.append(seg_npy_name)

        # -- prepare hr images --
        # print(len(glob.glob(os.path.join(img_dir, "*.npy"))),len(self.img_filenames))
        if len(glob.glob(os.path.join(img_dir, "*.npy"))) != len(self.img_filenames):
            for i in range(LEN):
                if (i+1) % 50 == 0:
                    print("convert {} hr images to npy data!".format(i+1))
                # print(self.hr_filenames)
                img_image = imageio.imread(self.img_filenames[i], pilmode="RGB")
                if self.colors == 1:
                    img_image = sc.rgb2ycbcr(img_image)[:, :, 0:1]
                img_fn_i = self.img_filenames[i]
                img_npy_name = img_fn_i.split('/')[-1].replace('.jpg', '.npy')
                img_npy_name = os.path.join(img_dir, img_npy_name)
                self.img_npy_names.append(img_npy_name)
                np.save(img_npy_name, img_image)
        else:
            pass
            # print("hr npy datas have already been prepared!, hr: {}".\
            #       format(len(self.hr_npy_names)))

        ## -- prepare seg images --
        if len(glob.glob(os.path.join(seg_dir, "*.npy"))) != len(self.seg_filenames):
            for i in range(LEN):
                if (i+1) % 50 == 0:
                    print("convert {} seg images to npy data!".format(i+1))
                # seg_image = imageio.imread(self.seg_filenames[i])#, pilmode="RGB")
                annos = loadmat(self.seg_filenames[i])['groundTruth']
                seg_image = annos[0][0]['Segmentation'][0][0]

                if self.colors == 1:
                    seg_image = sc.rgb2ycbcr(seg_image)[:, :, 0:1]
                seg_fn_i = self.seg_filenames[i]
                seg_npy_name = seg_fn_i.split('/')[-1].replace('.mat', '.npy')
                seg_npy_name = os.path.join(seg_dir, seg_npy_name)
                self.seg_npy_names.append(seg_npy_name)
                np.save(seg_npy_name, seg_image)
        else:
            pass
            # print("lr npy datas have already been prepared!, lr: {}".\
            #       format(len(self.lr_npy_names)))

    def __len__(self):
        if self.train:
            return self.nums_trainset * self.data_repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        idx = idx % self.nums_trainset
        img = np.load(self.img_npy_names[idx])
        seg = np.load(self.seg_npy_names[idx])
        if self.train:
            ps = self.patch_size
            train_img_patch, train_seg_patch = utils.crop_patch(img, seg, ps, True)
            return train_img_patch, train_seg_patch
        img = utils.ndarray2tensor(img).contiguous()
        seg = th.from_numpy(1.*seg).float()
        return img, seg
