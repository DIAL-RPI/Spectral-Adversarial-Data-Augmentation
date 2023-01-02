from __future__ import print_function, absolute_import
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

from sklearn import manifold, datasets
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
sys.path.append('./')
sys.path.append('../')

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

# '''self defined'''
# from progress.bar import Bar
from dataset import *
from utils import *
from model import ConvNet

cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


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
parser.add_argument('--lr', default=1e-2, help='learning rate', type=str)
# parser.add_argument('--gamma', default=0.1, help='learning rate decay', type=str)
parser.add_argument('--momentum', default=0.9, help='momentum', type=str)
parser.add_argument('--weight_decay', default=5e-4, help='weight decay', type=str)


### dataset
parser.add_argument('--data_path', 
                    default='put you dataset here', 
                    type=str)
parser.add_argument('--domain_name_src', default='MNIST', help='single domain', type=str)
parser.add_argument('--domain_name_tgt', default='MNIST', help='single domain', type=str)
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
args.epochs = 100

### USPS , mnist_m , SVHN , SYNTH
args.domain_name_src = 'MNIST'
args.domain_name_tgt = 'USPS'

args.batch_size = 128
args.phase_only = False

print(args)


'''dataset info'''
domain_name_dict = {'MNIST':1, 
                    'mnist_m':3, 
                    'USPS':3, 
                    'SVHN':3, 
                    'SYNTH':3}
transform_list = [
    transforms.Resize(32), 
    transforms.ToTensor(), # Turn PIL Image to torch.Tensaor
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
]

'''convert all images to 3-channel'''
channel_num = domain_name_dict[args.domain_name_src]
tf_test = transforms.Compose(transform_list) if channel_num==1 else transforms.Compose(transform_list[:-1])

'''generate dataloader'''
dl_test = testloader_generator(data_path=args.data_path,
                    domain_name=args.domain_name_src, 
                    batch_size=args.batch_size,
                    data_transform=tf_test,
                    is_shuffle=False, is_drop_last=False)
print("Len of {} Testset:{}".format(args.domain_name_src, len(dl_test)))

'''generate Fourier basis'''
A_basis_norm = get_basis(image_pix=32)
vis_basis(A_basis_norm)
print('ok')
print(A_basis_norm.shape)

domain_name = 'MNIST'
data_path = 'you source domain data location'
domain_name_dict = {'MNIST':1, 
                     'mnist_m':3, 
                     'USPS':3, 
                     'SVHN':3, 
                     'SYNTH':3}
transform_list = [transforms.Resize(32), 
                  transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                  transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                 ]

channel_num = domain_name_dict[domain_name]
tf_test = transforms.Compose(transform_list) if channel_num==1 else transforms.Compose(transform_list[:-1])
dataloader_MNIST = testloader_generator(data_path=data_path,
                                      domain_name=domain_name, 
                                      batch_size=64,
                                      data_transform=tf_test)

'''compute amplitude cube'''
Amp_tot, _ = Get_ds_all_AP_v1(dataloader=dataloader_MNIST, if_shift=True)
print(Amp_tot.shape)

'''compute average amplitude cube'''
weight_map = np.mean(Amp_tot,0)*1e1
print(weight_map.shape)

'''generate the mean source domain amplitude spectrum D'''
### resize even edge length to odd edge length
weight_map_ = np.concatenate((weight_map, weight_map[:,0].reshape(-1,1)), axis=1)
weight_map_ = np.concatenate((weight_map_, weight_map_[0,:].reshape(1,-1)), axis=0)
print(weight_map_.shape)
plt.imshow(weight_map_, cmap='jet')

plt.clim(0, 600)
plt.colorbar()

print(np.where(weight_map_==np.max(weight_map_)))
print(weight_map_.max(), weight_map_.min())

'''init model'''
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

# print('model domain name:{}'.format(args.domain_name))
print(args.save_dir)
model_path = os.path.join('put you source domain directory here')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])

'''generate and save enhanced model sensitivity map'''
save_map_dir = os.path.join(args.save_dir, 'ModelBias_WDwThld_ERM_org') # save directory
mkdir_if_missing(save_map_dir) 
FH_maps = eval_fourier_heatmap(
                    A_basis_norm=A_basis_norm,
                    dataloader=dl_test,
                    model=model,
                    perturb_radii=30,
                    r_factor=None,# rt_map_half, # None
                    if_save = True,
                    savedir = save_map_dir
                )

