import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

from models.id.networks.network import VGG
from utils import config
CONFIG = config.Config()

class FCLayers(nn.Module):
    def __init__(self, input_size=4096, z_size=2):
        super(FCLayers, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, z_size)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class VGG_Linear_Decoder(nn.Module):
    def __init__(self, z_size=None, output=None):
        super(VGG_Linear_Decoder, self).__init__()
        self.z_size = z_size
        self.output_level = output

        self.vgg_bfm_base = VGG(500)
        self.vgg_bfm_base.neural_test = False
        self.vgg_bfm_base.load_state_dict(torch.load(CONFIG['PATHS', 'checkpoints'], 'vgg', 'checkpoint_bfm.pth.tar')['state_dict'])

        self.module_1 = self.vgg_bfm_base.module_1
        self.module_2 = nn.Sequential(*list(self.vgg_bfm_base.module_2.children())[0:2])
        self.module_3 = nn.Sequential(*list(self.vgg_bfm_base.module_2.children())[2:])
        self.module_4 = self.vgg_bfm_base.module_3

        if output == 'tcl':
            self.light_depth = FCLayers(25088, self.z_size)
        else:
            self.light_depth = FCLayers(4096, self.z_size)

        for p,k in enumerate(self.module_1.parameters()):
            k.requires_grad = False
        for p,k in enumerate(self.module_2.parameters()):
            k.requires_grad = False
        for p,k in enumerate(self.module_3.parameters()):
            k.requires_grad = False
        for p,k in enumerate(self.module_4.parameters()):
            k.requires_grad = False

        self.module_1.eval()
        self.module_2.eval()
        self.module_3.eval()
        self.module_4.eval()

    def forward(self, x):

        mean = torch.FloatTensor([129.1863, 104.7624, 93.5940]).cuda()
        out_1 = self.module_1(x - mean.view(1, -1, 1, 1))
        out_2 = self.module_2(out_1)
        out_3 = self.module_3(out_2)
        out_4 = self.module_4(out_3)

        if self.output_level == None or self.output_level == 'sfcl':
            out = self.light_depth(out_4)
        elif self.output_level == 'tcl':
            out = self.light_depth(out_2)
        elif self.output_level == 'ffcl':
            out = self.light_depth(out_3)

        return out
