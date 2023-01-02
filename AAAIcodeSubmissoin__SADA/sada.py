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

def GetAPfromImage(image):
    img_fft = fft.fft2(image)
    img_fft_amp = torch.abs(img_fft)# 幅度谱
    img_fft_phase = torch.angle(img_fft)# 相位谱
    return img_fft_amp, img_fft_phase


def GetImagefromAP(amp, phase):
    img_recon = amp*torch.exp(1j*phase)
    img_recon = torch.abs(torch.fft.ifft2(img_recon))
    img_recon = img_recon/torch.max(img_recon)
    return img_recon

def attack_amp(model, image_origin, labels, config_attack):
    '''
    The implematation of early-stopped FourierAug
    Following the Alg.1 in our FAT paper <https://arxiv.org/abs/2002.11242>
    Args:
        step_size: the PGD step size
        epsilon: the perturbation bound
        perturb_steps: the maximum PGD step
        tau: the step controlling how early we should stop interations when wrong adv data is found
        randominit_type: To decide the type of random inirialization (random start for searching adv data)
        rand_init: To decide whether to initialize adversarial sample with random noise (random start for searching adv data)
        omega: random sample parameter for adv data generation (this is for escaping the local minimum.)
    Returns: 
        output_adv (friendly adversarial data) output_target (targets)
        output_natural (the corresponding natrual data)
    '''
    N, C, W, H = image_origin.shape
    image_origin, labels = image_origin.cuda(), labels.cuda()
    model = model.cuda()
    model.eval()
    budget = (torch.ones(len(labels)) * config_attack.tau).cuda()
    
    ### init weighting factor map
    if config_attack.weight_map is not None: 
        w_map = torch.Tensor(config_attack.weight_map).cuda()
    else:
        w_map = torch.Tensor([1]).cuda() #+ torch.rand(1).cuda()
    if config_attack.direction == 'pos':
        w_map = -w_map
        
    ### init loss function
    if config_attack.criterion == "ce":
        loss_fn = nn.CrossEntropyLoss(reduction='mean').cuda()
    elif config_attack.criterion == "kl":
        loss_fn = nn.KLDivLoss(size_average=False).cuda()
        # loss = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))

    ### get phase and amplitude
    with torch.no_grad():
        output_iter_clean_data = model(image_origin)
        amp_origin, phase_origin = GetAPfromImage(image_origin)
        amp, phase = amp_origin.detach(), phase_origin.detach()
    
    ### init perturb on amp
    if config_attack.random_init is True:
        if config_attack.randominit_type == "normal":
            amp[:,:,0,0] = (torch.rand_like(-amp[:,:,0,0])*4 + 1)*amp[:,:,0,0]
#             amp = (torch.randn_like(amp)*config_attack.epsilon + 1)*amp
        elif config_attack.randominit_type == "uniform":
            amp = amp + amp*torch.zeros_like(amp).uniform_(-config_attack.epsilon, config_attack.epsilon) # random init
        amp.detach_()
    amp.requires_grad = True
    phase.requires_grad = False
    
    for step in range(config_attack.num_steps):
        image_recon = GetImagefromAP(amp, phase)
        image_recon = torch.clamp(image_recon, 0.0, 1.0)
        output = model(image_recon)
        pred = output.max(1, keepdim=True)[1]
        
        ### Calculate the indexes of adversarial data those still needs to be iterated
        incorrect_sample_idx = (pred.squeeze() != labels)
        havent_used_up_budget_idx = (budget != 0)
        budget[torch.logical_and(incorrect_sample_idx, havent_used_up_budget_idx)] -= 1
        attack_sample_idx = (budget > 0).detach().cpu().numpy()
        if np.sum(attack_sample_idx) == 0: # all samples have used up their budgets
            break
        else:
            loss_adv = loss_fn(output, labels)
        grad = torch.autograd.grad(loss_adv, [amp])[0]
        grad.data[~attack_sample_idx].zero_()
        amp = amp.detach() + config_attack.step_size*torch.sign(grad.detach())*amp.detach()*w_map.detach()
        amp = torch.clamp(amp, 0.0)
        amp.requires_grad = True
        
    image_recon = GetImagefromAP(amp, phase)
    image_recon = torch.clamp(image_recon, 0.0, 1.0)
    image_recon = image_recon.detach()
    image_recon = TF.adjust_brightness(image_recon, 1.0+np.random.uniform(0,0.6))
    model.train() # turn on train model to do the domain and label calssification
    return image_recon
        


