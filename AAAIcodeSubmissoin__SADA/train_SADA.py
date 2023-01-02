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
from augmix import *
from sada import *
from train_test import *
from dataset import *
from model import ConvNet
from evaluation import eval_cross_domain

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


parser = argparse.ArgumentParser(description='RUN Baseline model of SADA !!!')

### dataset
parser.add_argument('--data_path', default='put you dataset here', type=str)
parser.add_argument('--domain_name_src', default='MNIST', help='single domain', type=str, 
                    choices=['MNIST', 'USPS', 'mnist_m', 'SVHN', 'SYNTH'])
parser.add_argument('--domain_name_tgt', default='USPS', help='single domain', type=str,
                    choices=['MNIST', 'USPS', 'mnist_m', 'SVHN', 'SYNTH'])
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--phase_only', default=False, help='generate phase-only image or not', type=bool)

### model training
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--classes', default=7, help='number of classes', type=int)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--lr', default=1e-1, help='learning rate', type=float)
# parser.add_argument('--gamma', default=0.1, help='learning rate decay', type=str)
parser.add_argument('--momentum', default=0.9, help='momentum', type=float)
parser.add_argument('--weight_decay', default=5e-4, help='weight decay', type=str)



### AugMix options
parser.add_argument('--consistency_reg', default=False, 
                    help='regularize the consistency between clean and adv images', type=bool)
parser.add_argument('--consistency_reg_weight', default=1.0, 
                    help='weight of consistency regularization', type=float)

parser.add_argument('--mixture_width', default=3,
                    type=int, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture_depth', default=-1,
                    type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug_severity', default=1,
                    type=int, help='Severity of base augmentation operators')
parser.add_argument('--aug_prob_coeff', default=1.,
                    type=float, help='Probability distribution coefficients')
parser.add_argument('--all_ops', default=False,
                    help='Turn on all operations (+brightness,contrast,color,sharpness).')



### save loc
parser.add_argument('--save_dir', 
                    default='./save_dir', 
                    type=str)

################### Initialize arguments ##################
args = parser.parse_args()
args.arch = 'ConvNet'
args.pretrained = False
args.start_epoch = 0
args.epochs = 50 # 200

args.lr = 0.01 # 0.1
args.consistency_reg = True
args.consistency_reg_weight = 0.25 #############################


args.domain_name = 'MNIST'
args.domain_name_tgt = 'mnist_m'
args.image_size = 32
args.batch_size = 256
args.phase_only = False
### compute the initial learning rate
# args.lr = args.lr * args.batch_size / 256
    
    
args.mixture_width = 3
args.mixture_depth = -1
args.aug_severity = 2
args.aug_prob_coeff = 1.0
args.all_ops = True


################# Prepare folders for saving ###############
_=mkdir_if_missing(args.save_dir)
if args.phase_only is True:
    args.save_dir = os.path.join(args.save_dir, 'Phase_only_'+args.domain_name)
else:
    args.save_dir = os.path.join(args.save_dir, 'FAug_FAug_FAug_DIGITS_'+args.arch+'_'+args.domain_name)
_=mkdir_if_missing(args.save_dir)

print('=========== settings of model training: ===========')
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))
    
    
    
################# SADA settings ########################
parser = argparse.ArgumentParser(description='Amp Attack Settings')

### model training
parser.add_argument('--epsilon',    default=0.2,  help='attack perturbation radii',       type=float)
parser.add_argument('--step_size',  default=0.08, help='attack steps size',               type=float)
parser.add_argument('--num_steps',  default=5,   help='number of attack steps',          type=int)
parser.add_argument('--tau',        default=1,    help='Early stop iteration for FAT.',   type=int)

parser.add_argument('--random_init', default=True, help='if init with random perturb', type=bool)
parser.add_argument('--randominit_type', default='normal', help='type of random init type', type=str)
parser.add_argument('--criterion', default='ce', help='loss functions to attack', choices=['ce', 'kl'], type=str)
parser.add_argument('--direction', default='neg', help='neg: standard attack || pos:inverse attack', choices=['pos','neg'], type=str)

#################################
args_attack = parser.parse_args()

print('=========== settings of amp attack: ===========')
for arg in vars(args_attack):
    print('{}: {}'.format(arg, getattr(args_attack, arg)))
    
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]

augmenter = Augmenter()

'''dataset info'''
domain_name_dict = {'MNIST':1, 
                    'mnist_m':3, 
                    'USPS':3, 
                    'SVHN':3, 
                    'SYNTH':3}
preprocess = transforms.Compose([
    transforms.ToTensor(), # Turn PIL Image to torch.Tensor
])

tf_train = transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),# if args.color_jitter else IdentityTransform(),
            ToGrayScale(3), # if args.grey else IdentityTransform(),
            transforms.ToTensor(),        
            GreyToColor(),
            LaplacianOfGaussianFiltering(size=3, sigma=1.0, identity_prob=0.6), # if args.LoG else IdentityTransform(),
            transforms.Lambda(lambda x: torch.clamp(x, min=0, max=1)),
            transforms.ToPILImage()
        ])

tf_test = transforms.Compose([
    transforms.Resize(36),
    transforms.CenterCrop(32),
#     transforms.Resize(32),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(), # Turn PIL Image to torch.Tensor
])

### target domain
tf_test_tgt = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(), # Turn PIL Image to torch.Tensor
])


'''generate dataloader'''
ds_train = dataset_target = datasets.MNIST(
                                        root=args.data_path,
                                        train=True,
                                        transform=tf_train,
                                        download=True
                                        )
ds_train = AugMixDataset(ds_train, preprocess)
dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

dl_test = testloader_generator(data_path=args.data_path,
                                domain_name=args.domain_name_src, 
                                batch_size=args.batch_size,
                                data_transform=tf_test,
                                is_shuffle=False, is_drop_last=False)

dl_test_tgt = testloader_generator(data_path=args.data_path,
                                domain_name=args.domain_name_tgt, 
                                batch_size=args.batch_size,
                                data_transform=tf_test_tgt,
                                is_shuffle=False, is_drop_last=False)

print("Len of {} Trainset:{}".format(args.domain_name_src, len(dl_train)))
print("Len of {} Testset:{}".format(args.domain_name_src, len(dl_test)))
print("Len of {} Testset:{}".format(args.domain_name_tgt, len(dl_test_tgt)))

### init model
if args.arch == 'resnet18':
    model = resnet18(pretrained=args.pretrained).cuda()
    model.fc = nn.Linear(model.fc.in_features, out_features=args.classes).cuda()
    model = torch.nn.DataParallel(model).cuda()
elif args.arch == 'ConvNet':
    model = ConvNet()
    model = torch.nn.DataParallel(model).cuda()

cudnn.benchmark = True
print('---> Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))



'''define optimizer and criterion'''
criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


'''Decay the learning rate scheduler'''
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 60], gamma=0.1)

'''load fourier heatmap'''
save_map_dir = 'model sensitivity map directory'
print('---> Loading parameters from {} domain'.format(args.domain_name_src))
image_path = os.path.join(save_map_dir, args.domain_name_src+'_only_'+"_fhmap_data.pth")
FH_maps = torch.load(image_path)
weight_map = FH_maps.numpy()+1

global best_acc
best_acc = 0  # best test accuracy

for epoch in range(args.start_epoch, args.epochs):
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[-1]['lr']))
    
    '''---Train---'''
    warm_up = True if epoch<=5 else False
    _, _, train_acc1_ema, batch_ema = train(net=model, 
                                warm_up=warm_up,
                                criterion=criterion, 
                                train_loader=dl_train, 
                                optimizer=optimizer,
                                args=args, args_attack=args_attack)
        
    '''---Test---'''
    _, test_acc = test(val_loader=dl_test, model=model, criterion=criterion)
    print('Source domain acc: {}'.format(test_acc))
    test_loss, test_acc_tgt = test(val_loader=dl_test_tgt, model=model, criterion=criterion)
    print('Target domain acc: {}'.format(test_acc_tgt))
#     scheduler.step()
    # save model if better than best
    is_best = test_acc > best_acc*0.95
    if is_best:
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=args.save_dir, filename='checkpoint_epoch_'+str(epoch)+'_cp1.pth.tar')
        test_loss, test_acc_tgt = test(val_loader=dl_test_tgt, model=model, criterion=criterion)
        print('===============Target domain acc: {}================'.format(test_acc_tgt))
    scheduler.step()
print('Best src acc: {}, Best tgt acc: {}'.format(best_acc, test_acc_tgt))


'''single-DG evaluation'''
model_path = os.path.join(args.save_dir, 'FAug_FAug_FAug_DIGITS_'+args.arch+'_'+args.domain_name, 'model_best.pth.tar')
criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

eval_cross_domain(data_path=args.data_path,
            model_path=model_path, 
            model_domain=args.domain_name,
            criterion=criterion,
            args=args)