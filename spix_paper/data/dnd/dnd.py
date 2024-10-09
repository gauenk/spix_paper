# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates a Dataset of unprocessed images for denoising.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as tf
# from unprocessing import unprocess

import torch as th
from .dnd_utils import unprocess


def load_dnd(args,load_test=False):
    data_path = optional(args,'data_path','./data')
    num_workers = optional(args,'num_workers',4)
    batch_size = optional(args,'batch_size',1)
    args = edict(get_fxn_kwargs(args,BSD500Seg.__init__)) # fill defaults
    dpath = Path(data_path)/'bsd500'
    cpath = dpath / ".cache"

    if load_test:
        dset = DNDDataset(dpath,cpath,
                         "test",data_augment=args.data_augment,colors=args.colors,
                         patch_size=args.patch_size,data_repeat=args.data_repeat)
        test_dataloader = DataLoader(dataset=dset, batch_size=1, shuffle=False)
        return dset,test_dataloader

    dset = DNDDataset(train=True)
    train_dataloader = DataLoader(dataset=dset,
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train,
                                  pin_memory=True, drop_last=True)
    dset = DNDDataset(train=True)
    valid_dataloader = DataLoader(dataset=dset,
                                  num_workers=num_workers,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train,
                                  pin_memory=True, drop_last=True)

    return train_dataloader,train_dataloader


class DNDDataset(data.Dataset):

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

def misc(data_dir,image_index,box_index):
    nimages = 50
    nboxes = 20

    # Loads image information and bounding boxes.
    info = h5py.File(os.path.join(data_dir, 'info.mat'), 'r')['info']
    bb = info['boundingboxes']

    # Loads the noisy image.
    filename = os.path.join(data_dir, 'images_raw', '%04d.mat' % (image_index + 1))
    img = h5py.File(filename, 'r')
    # print(list(img.keys()))
    noisy = np.float32(np.array(img['Inoisy']).T)

    # Loads raw Bayer color pattern.
    bayer_pattern = np.asarray(info[info['camera'][0][i]]['pattern']).tolist()

    # Denoises each bounding box in this image.
    boxes = np.array(info[bb[0][image_index]]).T

    # Loads shot and read noise factors.
    nlf_h5 = info[info['nlf'][0][image_index]]
    shot_noise = nlf_h5['a'][0][0]
    read_noise = nlf_h5['b'][0][0]

    # Read channels
    channels = read_dnd(noisy,bayer_pattern,boxes,shot_noise,read_noise,k)


def read_jpg(filename):
    """Reads an 8-bit JPG file from disk and normalizes to [0, 1]."""
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, channels=3)
    white_level = 255.0
    return tf.cast(image, tf.float32) / white_level


def is_large_enough(image, height, width):
    """Checks if `image` is at least as large as `height` by `width`."""
    image.shape.assert_has_rank(3)
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    return tf.logical_and(
        tf.greater_equal(image_height, height),
        tf.greater_equal(image_width, width))


def augment(image, height, width):
    """Randomly flips and crops `images` to `height` by `width`."""
    size = [height, width, tf.shape(image)[-1]]
    image = tf.random_crop(image, size)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def create_example(image):
    """Creates training example of inputs and labels from `image`."""
    image.shape.assert_is_compatible_with([None, None, 3])
    image, metadata = unprocess.unprocess(image)
    shot_noise, read_noise = unprocess.random_noise_levels()
    noisy_img = unprocess.add_noise(image, shot_noise, read_noise)
    # Approximation of variance is calculated using noisy image (rather than clean
    # image), since that is what will be avaiable during evaluation.
    variance = shot_noise * noisy_img + read_noise

    inputs = {
        'noisy_img': noisy_img,
        'variance': variance,
    }
    inputs.update(metadata)
    labels = image
    return inputs, labels


def create_dataset_fn(dir_pattern, height, width, batch_size):
    """Wrapper for creating a dataset function for unprocessing.
  
    Args:
      dir_pattern: A string representing source data directory glob.
      height: Height to crop images.
      width: Width to crop images.
      batch_size: Number of training examples per batch.

    Returns:
      Nullary function that returns a Dataset.
    """
    if height % 16 != 0 or width % 16 != 0:
      raise ValueError('`height` and `width` must be multiples of 16.')

    def dataset_fn_():
      """Creates a Dataset for unprocessing training."""
      autotune = tf.data.experimental.AUTOTUNE

      filenames = tf.data.Dataset.list_files(dir_pattern, shuffle=True).repeat()
      images = filenames.map(read_jpg, num_parallel_calls=autotune)
      images = images.filter(lambda x: is_large_enough(x, height, width))
      images = images.map(
          lambda x: augment(x, height, width), num_parallel_calls=autotune)
      examples = images.map(create_example, num_parallel_calls=autotune)
      return examples.batch(batch_size).prefetch(autotune)

    return dataset_fn_
