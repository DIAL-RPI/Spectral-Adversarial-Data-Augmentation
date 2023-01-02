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
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torchvision.utils import make_grid

'''self defined'''
# from progress.bar import Bar
from model import ConvNet
from utils import *
from dataset import *
from train_test_ERM import *

cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(torch.__version__)

def mkdir_if_missing(save_dir):
    if os.path.exists(save_dir):
        return 1
    else:
        os.makedirs(save_dir)
        return 0
    
cudnn.benchmark = True
manual_seed = 41
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)



parser = argparse.ArgumentParser(description='RUN Baseline model of vvv')

### model training
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--classes', default=7, help='number of classes', type=int)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--lr', default=1e-1, help='learning rate', type=str)
# parser.add_argument('--gamma', default=0.1, help='learning rate decay', type=str)
parser.add_argument('--momentum', default=0.9, help='momentum', type=str)
parser.add_argument('--weight_decay', default=5e-4, help='weight decay', type=str)


### dataset
parser.add_argument('--data_path', default='put your dataset here', type=str)
parser.add_argument('--domain_name', default='MNIST', help='single domain', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--phase_only', default=False, help='use use phase information', type=bool)

### save loc
parser.add_argument('--save_dir', 
                    default='./save_dir', 
                    type=str)


#################################
args = parser.parse_args()
args.model_name = 'ConvNet'
args.pretrained = True
args.start_epoch = 0
args.lr=0.01
args.epochs = 50

args.domain_name = 'MNIST'
args.batch_size = 256
args.phase_only = False

_=mkdir_if_missing(args.save_dir)
if args.phase_only is True:
    args.save_dir = os.path.join(args.save_dir, args.domain_name+'_phase_only')
else:
    args.save_dir = os.path.join(args.save_dir, args.domain_name)
_=mkdir_if_missing(args.save_dir)

print(args)

# NUM_CLASSES = 7      # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
# DATASETS_NAMES = ['photo', 'art', 'cartoon', 'sketch']
# CLASSES_NAMES = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']

### means and standard deviations ImageNet because the network is pretrained
means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
augmenter = Augmenter()

### transformations
tf_train = transforms.Compose([
#                                transforms.RandomResizedCrop(32),
                               transforms.Resize(32),      # Resizes short size of the PIL image to 256
#                                transforms.RandomCrop(32),
#                                transforms.CenterCrop(32),  # Crops a central square patch of the image 224
                               transforms.Lambda(augmenter),
                               transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                               transforms.Lambda(lambda x: x.repeat(3, 1, 1))
#                              transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
])
tf_test = transforms.Compose([
#                               transforms.Resize(36),      # Resizes short size of the PIL image to 256
                              transforms.CenterCrop(32),  # Crops a central square patch of the image 224
                              transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                              transforms.Lambda(lambda x: x.repeat(3, 1, 1))
#                              transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
])


### datasets
# dir_train = os.path.join(args.data_path, args.domain_name)
# dir_test = os.path.join(args.data_path, args.domain_name)

### datasets
ds_train = datasets.MNIST(root=args.data_path, train=True, transform=tf_train, download=True)
ds_test = datasets.MNIST(root=args.data_path, train=False, transform=tf_test, download=True)

# ds_train = torchvision.datasets.ImageFolder(dir_train, transform=tf_train)
# ds_test = torchvision.datasets.ImageFolder(dir_test, transform=tf_test)

### dataloaders
dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)

# Check dataset sizes
print("Len of {} Trainset:{}".format(args.domain_name, len(dl_train)))
print("Len of {} Testset:{}".format(args.domain_name, len(dl_test)))


### init model
if args.model_name == 'resnet18':
    model = resnet18(pretrained=args.pretrained).cuda()
    ### output (1000) --> output (7)
    model.fc = nn.Linear(model.fc.in_features, out_features=args.classes).cuda()
    model = torch.nn.DataParallel(model).cuda()
elif args.model_name == 'ConvNet':
    model = ConvNet()
    model = torch.nn.DataParallel(model).cuda()
    
cudnn.benchmark = True
print('---> Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


### define loss function (criterion) and optimize
criterion = nn.CrossEntropyLoss(reduction='none').cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

global best_acc
best_acc = 0  # best test accuracy

# Train and val
# writer = SummaryWriter(log_dir=args['checkpoint'])
# warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * args['warm'], start_lr=args['warm_lr']) if args['warm'] > 0 else None

for epoch in range(args.start_epoch, args.epochs):
#     if epoch >= args['warm'] and args['lr_schedule'] == 'step':
#         adjust_learning_rate(optimizer, epoch, args)
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[-1]['lr']))
    
    '''---Train---'''
    train(train_loader=dl_train, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, phase_only=args.phase_only)
        
    '''---Test---'''
    _, test_acc = test(val_loader=dl_test, model=model, criterion=criterion, epoch=epoch, phase_only=args.phase_only)
    # save model if better than best
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.save_dir, filename='checkpoint_epoch_'+str(epoch)+'.pth.tar')
#     if args.phase_only is False: scheduler.step()
    scheduler.step()
    
    ###
    print('Best acc: {}'.format(best_acc))
