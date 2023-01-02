from __future__ import print_function, absolute_import
import random
import os, sys, argparse, time
import math
import cv2
import argparse
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

def mkdir_if_missing(save_dir):
    if os.path.exists(save_dir):
        return 1
    else:
        os.makedirs(save_dir)
        return 0

    
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    '''
    '''
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

        
def img_norm_01(image):
    image = image-np.min(image)
    image = image/np.max(image)
    return image


def accuracy(output, target, topk=(1,), cls='all'):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def GetAPfromImage(image):
    img_fft = fft.fft2(image)
    img_fft_amp = torch.abs(img_fft)# 幅度谱
    img_fft_phase = torch.angle(img_fft)# 相位谱
    return img_fft_amp, img_fft_phase


def GetImagefromAP(amp, phase):
    img_recon = amp*torch.exp(1j*phase)
    img_recon = torch.abs(torch.fft.ifft2(img_recon))
    img_recon =img_recon/torch.max(img_recon)
    return img_recon

