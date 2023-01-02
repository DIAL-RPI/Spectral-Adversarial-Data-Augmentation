from __future__ import print_function, absolute_import
import random
import os, sys, argparse, time
import math
import cv2
import argparse
import numpy as np
import shutil
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



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x, return_feat=False):
        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if return_feat:
            return out4, self.fc3(out4)
        else:
            return self.fc3(out4)
