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

def get_basis(image_pix, dif_map=None):
    '''Generate all basis
    Args:
       image_pix(torch.Tensor): input image edge size
    Returns:
       A_basis_norm(torch.Tensor): Fourier basis in image space, [bs=(image_pix*H) , 1 , image_pix , image_pix]
    Notes: 
       No normalization !!!
    '''
    half_edge = int(image_pix/2)
    H = half_edge + 1 #17
    b_s = image_pix * H
    half_b_s = int(b_s/2)
    chn = 1
    index = 0
    
    if dif_map is not None:
        dis_map_1, dis_map_2 = dif_map.shape
        assert dis_map_1 == image_pix
        assert dis_map_2 == H
        dif_map = torch.Tensor(dif_map)
        
    A_basis_norm = torch.zeros(b_s, chn, image_pix, image_pix)
    for ii in np.arange(image_pix):
        for jj in np.arange(H):
            A_spectrum = torch.zeros(1, chn, image_pix, H)
            A_spectrum[:, :, ii, jj] = 1 if dif_map is None else dif_map[ii, jj]
            A_basis = fft.irfft2(A_spectrum)
            '''Note: no normalization !!!'''
            A_basis_norm[index] = A_basis
#             A_basis_norm[index] = A_basis / A_basis.norm(dim=(-2, -1))[None, None] if dif_map is None else A_basis #
#             A_basis_norm = F.normalize(A_basis.view(b_s, chn, -1),p=2.0,dim=2).view(b_s, chn, image_pix, image_pix).cuda() #
            index += 1
    ### switch the order (like the fft-shift)
    hold_ = torch.clone(A_basis_norm[:H*half_edge, :,:,:])
    A_basis_norm[:H*half_edge, :,:,:] = A_basis_norm[H*half_edge:, :,:,:]
    A_basis_norm[H*half_edge:] = hold_
    return A_basis_norm


def vis_basis(basis, if_save=False):
    '''visualize all basis in a single image
    '''
    _,_,image_pix,_ = basis.shape
    half_edge = int(image_pix/2)
    H = half_edge + 1 #17
    A_basis_norm = basis
    A_basis_norm = (A_basis_norm - A_basis_norm.min()) / (A_basis_norm.max()-A_basis_norm.min())

    # print images
    plt.figure(figsize=(30,30))
    img = make_grid(A_basis_norm[:H*32], nrow=half_edge+1, padding=1)

    # img = make_grid((A_basis_norm[0:14]-A_basis_norm.min())/(A_basis_norm.max()-A_basis_norm.min()), nrow=7, padding=2)
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    if if_save:
        torchvision.utils.save_image(basis, "./basis.png", nrow=H)
    
    
def get_perturb_image(data_batch, F_basis, epsilon=1.0):
    '''Add a single Fourier basis to image batch
    Args:
       data_batch(torch.Tensor): input image batch, [bat_size * channel * image_size * image_size]
       F_basis(torch.Tensor): input Fourier basis, [1 , 1 , image_size , image_size]
       epsilon(float): a multiplier to the original F_basis spetrum
    '''
    bs, chn, _, image_pix = data_batch.shape
    direction = [-1, 1]
    rand_num = bs * chn # random direction = batch size * channel num
    rand_dirct = torch.Tensor(random.choices(direction, k=rand_num)).float().view(bs, chn).cuda()
    rand_dirct.unsqueeze_(-1).unsqueeze_(-1)
    
    F_basis_norm = F_basis.norm(dim=(-2, -1))
    print('Basis norm: ', F_basis_norm)
#     print('F_basis_norm: ', F_basis_norm)
#     print('F_basis_norm x epsilon: ', F_basis_norm*epsilon)
    if F_basis_norm*epsilon <= 8: #17.5:
        F_basis = F_basis / F_basis_norm
        delta = F_basis.repeat(bs, chn, 1, 1).cuda()
        delta = rand_dirct * delta * 8 #17.5
        data_batch_new = data_batch.detach() + delta
    else:
        delta = F_basis.repeat(bs, chn, 1, 1).cuda()
        epsilon = torch.Tensor([epsilon]).cuda()#.repeat(bs, chn, 1, 1)
        delta = rand_dirct * delta * epsilon
        data_batch_new = data_batch.detach() + delta
#     data_batch_new = torch.clamp(data_batch_new, min=0, max=1)
    return data_batch_new


def single_freq_sens_eval_v2(val_loader, model, F_basis=None, epsilon=0.3):
    '''Measure model performance with single Fourier basis perturbation
    Args:
        val_loader (torch.utils.data.DataLoader): dataloader.
        model (nn.Module): a model to be evaluated.
        F_basis (torch.Tensor): a single Fourier basis map, [1 , 1 , image_size , image_size].
    Returns:
        top1.avg (numpy.array): model prediction accuracy, [1].
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    
    model = model.eval()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        ### measure data loading time
        data_time.update(time.time() - end)
        if F_basis is not None:
            inputs_perturb = get_perturb_image(data_batch=inputs, F_basis=F_basis, epsilon=epsilon)
            inputs_perturb = inputs_perturb.cuda()
        else:
            inputs_perturb = inputs
        
        ### compute output
        with torch.no_grad():
            outputs = model(inputs_perturb)
            
        ### measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        top1.update(prec1.item(), inputs.size(0))
        ### measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return top1.avg


def create_fourier_heatmap_from_error_matrix(
    error_matrix: torch.Tensor,
) -> torch.Tensor:
    """Create Full map from HALF map. Fourier Heat Map is symmetric about the origin.     
    Args:
        error_matrix (torch.Tensor): The size of error matrix should be (H, H/2+1). Here, H is height of image.
                                     This error matrix shoud be about quadrant 1 and 4.
    Returns:
        torch.Tensor (torch.Tensor): Fourier Heat Map created from error matrix.
    """
    assert len(error_matrix.size()) == 2
    assert error_matrix.size(0) == 2 * (error_matrix.size(1) - 1)

    fhmap_rightside = error_matrix[1:, :-1]
    fhmap_leftside = torch.flip(fhmap_rightside, (0, 1))
    return torch.cat([fhmap_leftside[:, :-1], fhmap_rightside], dim=1)

def save_fourier_heatmap(fhmap, savedir):
    """Save Fourier Heat Map as a png image.
    Args:
        fhmap (torch.Tensor): Fourier Heat Map. [image_pix-1 , image_pix-1]
        savedir (string): Path to the directory where the results will be saved.
    """
    _ = mkdir_if_missing(savedir)
    torch.save(fhmap, os.path.join(savedir, args.domain_name_src+'_only_'+"_fhmap_data.pth"))
    sns.heatmap(
        fhmap.numpy(),
        vmin=0.0,
        vmax=1.0,
        cmap="jet",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.savefig(os.path.join(savedir, args.domain_name_src+'_only_'+"_fhmap_data.png"))
    plt.close("all") 
    
    
def eval_fourier_heatmap(
    A_basis_norm,
    dataloader,
    model,
    perturb_radii,
    r_factor=None,
    if_save = False,
    savedir = None
):
    '''Evaluate Fourier Heat Map about given architecture and dataset.
    Args:
        A_basis_norm(torch.Tensor): a series of Fourier basis, [(image_pix*H) , 1 , image_pix , image_pix].
        val_loader (torch.utils.data.DataLoader): dataloader.
        model (nn.Module): a model to be evaluated.
        perturb_radii: the amplitude of the Fourier basis in the image space.
        if_save: decided if you want to save the Fourier heat map.
        savedir (string): Path to the directory where the results will be saved.
    If Save:
        fourier_heatmap(torch.Tensor): [image_pix-1 , image_pix-1]
    Return:
        fourier_heatmap(torch.Tensor): [image_pix-1 , image_pix-1]
    '''
    _ = mkdir_if_missing(savedir)
    acc = []
    _,_,image_pix,_ = A_basis_norm.shape
    H = int(image_pix/2) + 1
    num_of_basis = len(A_basis_norm)
    
    for ii in np.arange(num_of_basis): ### Go through all Fourier basis
        single_basis = A_basis_norm[ii].unsqueeze_(0) #take one basis out [1 , 1 , image_pix , image_pix]
        if r_factor is not None:
#             perturb_radii_new = perturb_radii * (r_factor[ii]+1)
            perturb_radii_new = perturb_radii * r_factor[ii]
            acc_ = single_freq_sens_eval_v2(val_loader=dataloader, model=model, F_basis=single_basis, epsilon=perturb_radii_new)
            print('ii={}, r_factor={}, perturb_radii={}, err={}'.format(ii, r_factor[ii], perturb_radii_new, 100-acc_))
        else:
            acc_ = single_freq_sens_eval_v2(val_loader=dataloader, model=model, F_basis=single_basis, epsilon=perturb_radii)
            print('ii={}, r_factor={}, err={}'.format(ii, 0, 100-acc_))
        acc.append(acc_)
    acc_map = np.asarray(acc)
    err_map = 100 - torch.Tensor(acc_map.reshape(image_pix,H))
    fourier_heatmap = create_fourier_heatmap_from_error_matrix(err_map)

    if if_save is True and savedir is not None: 
        save_fourier_heatmap(fourier_heatmap/100.0, savedir)
    return fourier_heatmap


def Get_ds_all_AP_v1(dataloader, if_shift=True):
    '''Get all dataset amplitude and phase from dataloader
    Args:
        dataloader: [torch.utils.data.DataLoader] 
        if_shift: [bool] shift the original point (0,0) to the image center
    Returns:
        A_tot: [torch.Tensor] dataset amplitude spectrum cube (N, C, H, W)
        P_tot: [torch.Tensor] dataset phase spectrum cube (N, C, H, W)
    '''
    dataloader_it = iter(dataloader)
    dataloader_len = len(dataloader)
    total_size = dataloader_len * dataloader.batch_size
    ### get the input image size
    transforms_ = dataloader.dataset.transform.transforms
    for index, tf_item in enumerate(transforms_):
        if isinstance(tf_item, torchvision.transforms.Resize):
            image_size = tf_item.size
    ### fill in AP 
    print('total_size: {}, image_size: {}'.format(total_size, image_size))
    A_tot, P_tot = np.zeros([total_size,image_size,image_size]), np.zeros([total_size,image_size,image_size])
    for ii in np.arange(dataloader_len):
        batch_datas, _ = next(dataloader_it)
        A, P = GetAPfromImage(batch_datas)
        if if_shift:
            A, P = torch.fft.fftshift(A), torch.fft.fftshift(P)
        
        left_, right_ = ii*dataloader.batch_size, (ii+1)*dataloader.batch_size
        A_tot[left_:right_, :,:] = A.cpu().data.numpy()[:, 0, :,:]
        P_tot[left_:right_, :,:] = P.cpu().data.numpy()[:, 0, :,:]
    return A_tot, P_tot

