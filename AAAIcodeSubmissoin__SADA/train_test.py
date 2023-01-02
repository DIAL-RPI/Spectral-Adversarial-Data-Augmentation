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

'''self defined'''
# from progress.bar import Bar
from transforms import GreyToColor, IdentityTransform, ToGrayScale, LaplacianOfGaussianFiltering


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
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
        
        
def mkdir_if_missing(save_dir):
    if os.path.exists(save_dir):
        return 1
    else:
        os.makedirs(save_dir)
        return 0

    
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
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


def compute_mce(corruption_accs):
    """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
    mce = 0.
    for i in range(len(CORRUPTIONS)):
        avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
        ce = 100 * avg_err / ALEXNET_ERR[i]
        mce += ce / 15
    return mce


def train(net, warm_up, criterion, train_loader, optimizer, args, args_attack):
    """Train for one epoch."""
    net.train()
    data_ema = 0.
    batch_ema = 0.
    loss_cln_ema = 0.
    loss_ctr_ema = 0.
    acc1_ema = 0.
    acc5_ema = 0.

    end = time.time()
    for batch_idx, (images, targets) in enumerate(train_loader):
        ### Compute data loading time
        data_time = time.time() - end
        optimizer.zero_grad()
        if args.consistency_reg is False:
            images = images.cuda()
            targets = targets.cuda()
            logits = net(images)
            loss_cln = criterion(logits, targets)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        if args.consistency_reg is True:
            # get augmix images
            img_clean, img_aug1, img_aug2, img_aug3 = images[0].cuda(), images[1].cuda(), images[2].cuda(), images[3].cuda()
            targets = targets.cuda()
            # get F-aug images
            if warm_up is False:
                img_aug1 = attack_amp(model=net,
                                      image_origin=img_aug1, 
                                      labels=targets, 
                                      config_attack=args_attack)
                img_aug2 = attack_amp(model=net,
                                      image_origin=img_aug2, 
                                      labels=targets, 
                                      config_attack=args_attack)
                img_aug3 = attack_amp(model=net,
                                      image_origin=img_aug3, 
                                      labels=targets, 
                                      config_attack=args_attack)
                args.consistency_reg_weight = 0.025
            else:
                args.consistency_reg_weight = 10
            # compute logits saperately
            logits_clean, logits_aug1, logits_aug2, logits_aug3 = net(img_clean), net(img_aug1), net(img_aug2), net(img_aug3)
            
            # Cross-entropy is only computed on clean images
            loss_cln = criterion(logits_clean, targets)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_clean, p_aug1 = F.softmax(logits_clean, dim=1), F.softmax(logits_aug1, dim=1)
            p_aug2, p_aug3 = F.softmax(logits_aug2, dim=1), F.softmax(logits_aug3, dim=1)
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2 + p_aug3) / 4., 1e-7, 1).log()
            loss_ctr = args.consistency_reg_weight * ( F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                                    F.kl_div(p_mixture, p_aug2, reduction='batchmean') +
                                                    F.kl_div(p_mixture, p_aug3, reduction='batchmean') ) / 4.
            loss = loss_ctr + loss_cln
            acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking
        loss.backward()
        optimizer.step()
        
        ### Compute batch computation time and update moving averages.
        batch_time = time.time() - end
        end = time.time()
        data_ema = data_ema * 0.1 + float(data_time) * 0.9
        batch_ema = batch_ema * 0.1 + float(batch_time) * 0.9
        loss_cln_ema = loss_cln_ema * 0.1 + float(loss_cln) * 0.9
        loss_ctr_ema = loss_ctr_ema * 0.1 + float(loss_ctr) * 0.9
        acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9
        acc5_ema = acc5_ema * 0.1 + float(acc5) * 0.9
        
#         if batch_idx % args.print_freq == 0:
        print(
            'Batch {}/{}:' 
            '| DataTime {:.3f} '
            '| BatchTime {:.3f} '
            '| Cln Loss {:.3f} '
            '| Ctr Loss {:.3f} '
            '| TrAcc1 {:.3f} '
            '| TrAcc5 {:.3f}'.format(
            batch_idx, len(train_loader), 
            data_ema,
            batch_ema, 
            loss_cln_ema, 
            loss_ctr_ema, 
            acc1_ema,
            acc5_ema))
    return loss_cln_ema, loss_ctr_ema, acc1_ema, batch_ema



def test(val_loader, model, criterion):
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.phase_only is True:
            A, P = GetAPfromImage(inputs)
            A_non = torch.zeros_like(A)+torch.mean(A)
            inputs = GetImagefromAP(A_non, P)
            inputs = 1-inputs/torch.max(inputs)
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        ### measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        ### measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses.avg, top1.avg)

