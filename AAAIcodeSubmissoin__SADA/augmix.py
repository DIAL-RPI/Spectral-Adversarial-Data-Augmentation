'''code adopted from: https://github.com/google-research/augmix'''
from __future__ import print_function, absolute_import
import random
import os, sys, argparse, time
import math
import cv2
import numpy as np
import shutil
sys.path.append('./')
sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import imgaug.augmenters as iaa
import imgaug as ia
from PIL import Image

from sklearn import manifold, datasets
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.autograd import Variable
import torch.fft as fft

import torchvision
import torchvision.models as models
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

"""Base augmentations operators."""
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
# IMAGE_SIZE = 32
def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((args.image_size, args.image_size),
                               Image.AFFINE, (1, level, 0, 0, 1, 0),
                               resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((args.image_size, args.image_size),
                               Image.AFFINE, (1, 0, 0, level, 1, 0),
                               resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), args.image_size / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((args.image_size, args.image_size),
                               Image.AFFINE, (1, 0, level, 0, 1, 0),
                               resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), args.image_size / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((args.image_size, args.image_size),
                               Image.AFFINE, (1, 0, 0, 0, 1, level),
                               resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


args.augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

# args.augmentations = [
#     rotate, shear_x, shear_y,
# #     translate_x, translate_y
# ]

args.augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]


def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
    Returns:
        mixed: Augmented and mixed image.
    """
    aug_list = args.augmentations
    if args.all_ops:
        aug_list = args.augmentations_all

    ws = np.float32(np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
    m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)
    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self, dataset, preprocess, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess), aug(x, self.preprocess), aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)