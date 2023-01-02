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
from utils import accuracy

def test(val_loader, model, 
         criterion,
         phase_only=False):
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
        
        if phase_only is True:
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



def eval_cross_domain(data_path, model_path, model_domain, criterion, args):
    ### transform
    tf_test_tgt = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(), # Turn PIL Image to torch.Tensor
    ])
    ### load the model
    print('---------------- loading model ----------------')
    if args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained).cuda()
        model.fc = nn.Linear(model.fc.in_features, out_features=args.classes).cuda()
        model = torch.nn.DataParallel(model).cuda()
    elif args.arch == 'ConvNet':
        model = ConvNet()
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('---> Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('model domain name:{}'.format(args.domain_name_src))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

#     model_obj = models.__dict__[args.arch]
#     model = model_obj(pretrained=False).cuda()
#     model.fc = nn.Linear(model.fc.in_features, out_features=args.classes).cuda()
#     model = torch.nn.DataParallel(model).cuda()
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['state_dict'])
    
    domain_name_list = ['USPS', 'mnist_m', 'SVHN', 'SYNTH']
    for domain_name in domain_name_list:
#         dir_test = os.path.join(data_path, 'PACS_test', domain_name)
#         ds_test = torchvision.datasets.ImageFolder(dir_test, transform=tf_test_tgt)
#         dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
        dl_test = testloader_generator(data_path=args.data_path,
                                        domain_name=domain_name, 
                                        batch_size=args.batch_size,
                                        data_transform=tf_test_tgt,
                                        is_shuffle=False, is_drop_last=False)
        print('======{} domain ======='.format(domain_name))
        _, acc = test(dl_test, model, criterion)
        print('original dataset: top1_acc: {}'.format(acc))
