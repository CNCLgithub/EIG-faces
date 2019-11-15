import numpy as np
from scipy.ndimage.measurements import label
from skimage.transform import resize 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

import v1


class V1(nn.Module):
    def __init__(self, z_size=None):
        super(V1, self).__init__()
        self.vision_module = v1.v1
        self.vision_module.load_state_dict(torch.load('/om/user/ilkery/eig_faces/models/raw/v1.pth'))
        self.module_1 = nn.Sequential(*list(self.vision_module.children())[:-5]) #TCL
        self.module_2 = nn.Sequential(*list(self.vision_module.children())[-5:-2]) #FC 1
        self.module_3 = nn.Sequential(*list(self.vision_module.children())[-2:-1]) #FC 2

    def forward(self, x):

        mean = torch.FloatTensor([104.0510072177276,  112.51448910834733,  116.67603893449996]).cuda()
        out_1 = self.module_1(x - mean.view(1, -1, 1, 1))
        out_2 = self.module_2(out_1)
        out_3 = self.module_3(out_2)
        if self.neural_test == True:
            return out_1, out_2, out_3
        else:
            return out_3
