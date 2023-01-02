import torch.utils.data as data
from PIL import Image
import os
import time
import random
import os, sys
import math
import cv2
import argparse
import numpy as np
import shutil
import copy
import pathlib
# from tqdm import tqdm
from collections import OrderedDict
from typing import Final, cast, Optional, Tuple, List, Dict, Iterator

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

import imgaug.augmenters as iaa
import imgaug as ia
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.autograd import Variable
import torch.fft as fft

import torchvision
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torchvision.utils import make_grid

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data
    
    
class SynthDigit(data.Dataset):
    """A synthetic counterpart of the SVHN dataset used in the paper
    "Unsupervised Domain Adaptation by Backpropagation" by Yaroslav Ganin and Victor Lempitsky.
    Download from
        https://drive.google.com/file/d/0B9Z4d7lAwbnTSVR1dEFSRUFxOUU/view
        linked from Yaroslav's homepage http://yaroslav.ganin.net/
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN_PyTorch/master/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=False, transform=None, target_transform=None):
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "synth_train_32x32.mat" if train else "synth_test_32x32.mat"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError("Dataset not found at {}".format(
                os.path.join(self.root, self.filename)
            ))

        import scipy.io as sio
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()
        self.data = np.transpose(self.data, (3, 2, 0, 1))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))
    
    
### shared function for different datasets
def testloader_generator(data_path, domain_name, 
                         batch_size, data_transform,
                         is_shuffle=False, is_drop_last=True):
    '''
    Notes: we do not shuffle and drop the last batch by default
    '''
    data_dir = os.path.join(data_path, domain_name)
    '''init datasets'''
    if domain_name == 'MNIST':
        dataset_target = datasets.MNIST(
                                        root=data_path,
                                        train=False,
                                        transform=data_transform,
                                        download=True
                                        )
        
    elif domain_name == 'mnist_m':
        test_list = os.path.join(data_dir, 'mnist_m_test_labels.txt')
        dataset_target = GetLoader(
                                   data_root=os.path.join(data_dir, 'mnist_m_test'),
                                   data_list=test_list,
                                   transform=data_transform
                                   )
        
    elif domain_name == 'USPS':
        test_list = os.path.join(data_dir, 'USPStest_label.txt')
        dataset_target = GetLoader(
                                   data_root=os.path.join(data_dir, 'test'),
                                   data_list=test_list,
                                   transform=data_transform
                                   )
        
    elif domain_name == 'SVHN':
        dataset_target = datasets.SVHN(
                                       root=data_dir,
                                       split='test',
                                       transform=data_transform,
                                       download=True
                                       )
        
    elif domain_name == 'SYNTH':
        dataset_target = SynthDigit(root=data_dir, 
                                    train=False, 
                                    transform=data_transform
                                    )
        
    '''init dataloaders'''
    dataloader_target = torch.utils.data.DataLoader(
                                                    dataset=dataset_target,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=8, 
                                                    drop_last=True
                                                    )
    return dataloader_target
