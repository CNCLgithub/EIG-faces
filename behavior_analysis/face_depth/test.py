import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from models.eig.networks.network import EIG
from models.eig.networks.network_classifier import EIG_classifier
from models.id.networks.network_linear_decoder import VGG_Linear_Decoder

from PIL import Image
import h5py

from utils import config
CONFIG = config.Config()

def load_image(image, size):
    image = image.resize(size)
    image = np.asarray(image)
    if len(image.shape) == 2:
        image = np.stack((image,)*3, -1)
    else:
        image = image[:,:,0:3]
    image = np.moveaxis(image, 2, 0)
    image = image.astype(np.float32)
    return image

models_d = {
    'eig' : EIG(),
    'eig_classifier' : EIG_classifier(),
    'vgg' : lambda d: VGG_Linear_Decoder(1, d),
}

image_sizes = {
    'eig' : (227, 227),
    'eig_classifier' : (227, 227),
    'vgg' : (224, 224),
}

filenames_d = {
    'eig' : 'eig.hdf5',
    'eig_classifier' : 'eig_classifier.hdf5',
    'vgg' : 'vgg.hdf5',
}


def main():
    parser = argparse.ArgumentParser(description='Predictions of the models on the face depthjudgment task.')
    parser.add_argument('model', type=str, 
                        help='Which model is being tested: eig, eig_classifier, vgg')
    parser.add_argument('--resume', type = str, default='', 
                        help='Where is the model weights stored if other than where the configuration specifies.')
    parser.add_argument('--level', type = str, default = 'ffcl', 
                        help = 'Required only for vgg. Layer at which to decode from the network: sfcl, ffcl, tcl')

    global args
    args = parser.parse_args()
    # This is where figures and network outputs go.
    if not os.path.exists('./output'):
        os.mkdir('./output')
    outfile = './output/' + filenames_d[args.model]

    print("=> Construct the model...")
    if args.model == 'vgg':
        model = models_d[args.model](args.level)
    else:
        model = models_d[args.model]
    model.cuda()
    model.eval()

    resume_path = args.resume
    if resume_path == '':
        if args.model == 'vgg':
            resume_path = os.path.join(CONFIG['PATHS', 'checkpoints'], args.model, 'depth', args.level, 'checkpoint_bfm.pth.tar')
        else:
            resume_path = os.path.join(CONFIG['PATHS', 'checkpoints'], args.model, 'checkpoint_bfm.pth.tar')
    checkpoint = torch.load(resume_path)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume_path, checkpoint['epoch']))
    
    test(model, outfile)

def test(model, outfile):
    path = os.path.join(CONFIG['PATHS', 'behavior'], 'stimuli/face_depth')

    N = 6
    dtype = torch.FloatTensor

    z = []
    counter = 0
    for j in range(N):
        for i in range(9):
            counter += 1
            for k in ['regular', 'relief']:

                fname = os.path.join(path, str(counter) + '_' + str(i+1) + '_' + k + '.png')
                v = Image.open(fname)
                image = load_image(v, image_sizes[args.model])
                image = torch.from_numpy(image).type(dtype).cuda()
                image = image.unsqueeze(0)

                if args.model == 'vgg':
                    out = model(image)
                else:
                    _, _, _, out = model(image, test=True)
                    out = out/10.

                out = out.detach()[0].cpu().numpy()
                z.append(out.flatten())

    f = h5py.File(outfile, 'w')
    f.create_dataset(str(0), data=np.array(z))
    f.close()

if __name__ == '__main__':
    main()


