import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

from models.raw import vgg_face_caffe
from models.raw import places365_alexnet_float

from utils import config
CONFIG = config.Config()

class FCLayers(nn.Module):
    def __init__(self, input_size, z_size):
        super(FCLayers, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, z_size)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class VGG(nn.Module):
    def __init__(self, z_size=None):
        super(VGG, self).__init__()
        self.z_size = z_size
        self.all_modules = vgg_face_caffe.vgg_face_caffe
        self.all_modules.load_state_dict(torch.load(os.path.join(CONFIG['PATHS', 'rawnets'], 'vgg_face_caffe.pth')))
        self.module_1 = nn.Sequential(*list(self.all_modules.children())[:30]) #TCL
        self.module_2 = nn.Sequential(*list(self.all_modules.children())[30:34]) # FC 1 with dropout
        self.module_3 = nn.Sequential(*list(self.all_modules.children())[34:38]) # FC 2 with dropout

        if self.z_size is not None:
            self.module_4 = FCLayers(4096, self.z_size)
        else:
            self.module_4 = None

        for p,k in enumerate(self.module_1.parameters()):
            if p > 23:
                break
            k.requires_grad = False


    def forward(self, x, test=False):
        mean = torch.FloatTensor([129.1863, 104.7624, 93.5940]).cuda()
        out_1 = self.module_1(x - mean.view(1, -1, 1, 1))
        out_2 = self.module_2(out_1)
        out_3 = self.module_3(out_2)
        if self.z_size is not None:
            out_4 = self.module_4(out_3)

        if test == True:
            return out_1, out_2, out_3
        else:
            return out_4

