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

def train(train_loader, model, criterion, optimizer, epoch, phase_only=False):
    # switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    
#     bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)
        
        if phase_only is True:
            A, P = GetAPfromImage(inputs)
            A_non = torch.zeros_like(A)+torch.mean(A)
            inputs = GetImagefromAP(A_non, P)
            inputs = 1-inputs/torch.max(inputs)
        
        # optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        losses.update(loss.item(), outputs.size(0))
        loss_str = "{:.4f}".format(losses.avg)
#         top1_str = "{:.4f}".format(top1.avg)
        suffix = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.2f}s | Loss: {loss:s}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    loss=loss_str
                    )
        print(suffix)
#     bar.finish()
    
    
def test(val_loader, model, criterion, epoch, mode='normal', phase_only=False):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
#     bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # measure data loading time
        data_time.update(time.time() - end)
#         inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        if phase_only is True:
            A, P = GetAPfromImage(inputs)
            A_non = torch.zeros_like(A)+torch.mean(A)
            inputs = GetImagefromAP(A_non, P)
            inputs = 1-inputs/torch.max(inputs)
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)