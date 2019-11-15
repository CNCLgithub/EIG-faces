import numpy as np
from scipy.ndimage.measurements import label

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

from models.raw import vrn_unguided
from models.raw import places365_alexnet_float
from models.eig.networks.network_classifier import EIG_classifier

from utils import config

CONFIG = config.Config()

def bbox(img):
    rows = np.any(img.numpy(), axis=1)
    cols = np.any(img.numpy(), axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def reposition(cmin, cmax, offset):
    if cmax > 227 - offset:
        cmin = cmin - (cmax - (227 - offset))
        cmax = 227 - offset
    elif cmin < offset:
        cmax = cmax - cmin - offset
        cmin = offset
    
    return cmin, cmax


def get_padding_params(rmin, rmax, cmin, cmax, img, add_offset):

    mid_point = (rmax + rmin) / 2. * 227 / 192
    rmax = np.ceil(mid_point + img.shape[1]/2.)
    rmin = np.ceil(mid_point - img.shape[1]/2.)
    if rmax > 227:
        rmin = rmin - (rmax - 227)
        rmax = 227
    elif rmin < 0:
        rmax = rmax - rmin
        rmin = 0
    aux_a = rmin
    aux_b = 227 - img.shape[1] - aux_a

    mid_point = (cmax + cmin) / 2. * 227 / 192
    cmax = np.ceil(mid_point + img.shape[2]/2.)
    cmin = np.ceil(mid_point - img.shape[2]/2.)
    offset = 0
    if add_offset:
        offset = 25 # avoid borders of the image
        if img.shape[2] > 227 - 2 * offset:
            offset = 0
        else:
            cmin, cmax = reposition(cmin, cmax, offset)

    aux_c = cmin
    aux_d = 227 - img.shape[2] - aux_c

    return int(aux_a), int(aux_b), int(aux_c), int(aux_d)


def process_segmentation(vols, x, add_offset):
    dtype = torch.FloatTensor
    vols = (vols * 255).type(torch.uint8)

    vols = torch.sum(vols, 1)
    vols_0 = torch.zeros(vols.size()).type(dtype).cuda()
    vols_1 = torch.ones(vols.size()).type(dtype).cuda()
    vols = torch.where(vols <= 1, vols_0, vols_1)

    out_x = torch.zeros((x.size()[0], x.size()[1], 227, 227))

    for k in range(x.size()[0]):
        vol = vols[k]
        L, num = label(vol.cpu())
        L = torch.from_numpy(L)
        img = x[k]
        areas = torch.zeros(num)
        for m in range(num):
            areas[m] = torch.sum(L == (m + 1))

        region = torch.argmax(areas) + 1
        L[L != region.int()] = 0
        rmin, rmax, cmin, cmax = bbox(L)
        img[:, L == 0] = 1.
        img = img[:, rmin:rmax, cmin:cmax]
        scale = 227. / torch.max(torch.Tensor([rmax - rmin + 1, cmax - cmin + 1]))
        size_give = (int(np.floor(img.shape[1] * scale)), int(np.floor(img.shape[2] * scale)))
        img = nn.Upsample(size=size_give, mode='bilinear', align_corners=True)(img.unsqueeze(0)).squeeze()

        # padding
        aux_a, aux_b, aux_c, aux_d = get_padding_params(rmin, rmax, cmin, cmax, img, add_offset)
        #print([aux_a, aux_b, aux_c, aux_d])
        img = torch.nn.functional.pad(img, pad=((aux_c, aux_d, aux_a, aux_b, 0, 0)), 
                        mode='constant', value=1)
        out_x[k] = img

    return out_x.type(dtype).cuda()


class FCLayers(nn.Module):
    def __init__(self, input_size=4096, z_size=404):
        super(FCLayers, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, z_size)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class EIG_classifier_early_layers(nn.Module):
    def __init__(self, imageset='fiv'):
        super(EIG_classifier_early_layers, self).__init__()

        self.eig_classifier = EIG_classifier()
        self.eig_classifier.load_state_dict(torch.load(os.path.join(CONFIG['PATHS', 'checkpoints'], 'eig_classifier', 'checkpoint_' + imageset + '.pth.tar'))['state_dict'])

        self.vision_module = self.eig_classifier.vision_module
        self.vision_module_conv1 = nn.Sequential(*list(self.vision_module.children())[:4]) 
        self.vision_module_conv2 = nn.Sequential(*list(self.vision_module.children())[4:8]) 
        self.vision_module_conv3 = nn.Sequential(*list(self.vision_module.children())[8:10])
        self.vision_module_conv4 = nn.Sequential(*list(self.vision_module.children())[10:12]) 

    def forward(self, x, segment=False, add_offset=False):
        dtype = torch.FloatTensor        
        if segment == True:
            x = nn.Upsample(size=(192, 192), mode='bilinear', align_corners=True)(x)
            segment_vols = self.eig_classifier.segmentation(x/255.)[0]
            segmented = process_segmentation(segment_vols.detach(), x/255, add_offset)
            x = segmented * 255

        mean = torch.FloatTensor([104.0510072177276,  112.51448910834733,  116.67603893449996]).cuda()

        out_interim = []
        x = x - mean.view(1, -1, 1, 1)
        x = self.vision_module_conv1(x)
        out_interim.append(x.detach()[0].cpu().numpy().flatten())
        x = self.vision_module_conv2(x)
        out_interim.append(x.detach()[0].cpu().numpy().flatten())
        x = self.vision_module_conv3(x)
        out_interim.append(x.detach()[0].cpu().numpy().flatten())
        x = self.vision_module_conv4(x)
        out_interim.append(x.detach()[0].cpu().numpy().flatten())

        return out_interim

    
    
    
