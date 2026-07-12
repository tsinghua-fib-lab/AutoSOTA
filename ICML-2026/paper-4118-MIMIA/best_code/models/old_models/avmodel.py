import os
import sys
from PIL import Image
import torch
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
import argparse
import csv
import random
import warnings
import pdb
sys.path.append('/home/xiaokang_peng/ks/models')
import encodera as ma
import encoderv as mv
warnings.filterwarnings('ignore')



class AVmodel(nn.Module):
    def __init__(self,args):
        super(AVmodel,self).__init__()
        self.args = args
        self.parta = ma.Resnet(self.args)
        self.parta.fc = nn.Linear(512, args.n_classes)

        self.partv = mv.Resnet(self.args)
        self.partv.fc = nn.Linear(512, args.n_classes)

        self.fc_ = nn.Linear(1024, args.n_classes)
        
        self.dropx = nn.Dropout(0.0)
        self.dropy = nn.Dropout(0.5)



    def forward(self,audio,visual,label,iterations):

        y = self.parta(audio)
        x = self.partv(visual)
        (_, C, H, W) = x.size()
        B = y.size()[0]
        x = x.view(B, -1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = F.adaptive_avg_pool3d(x, 1)
        y = F.adaptive_avg_pool2d(y, 1)
        x = x.squeeze(2).squeeze(2).squeeze(2)
        y = y.squeeze(2).squeeze(2)
        
        #x = self.dropx(x)
        #y = self.dropy(y)
        #x *= self.dropx(torch.ones(1)).cuda()
        #y *= self.dropy(torch.ones(1)).cuda()

        out = torch.cat((x, y),1)
        out = self.fc_(out)
        

        return x, y, out

