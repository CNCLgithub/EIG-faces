import argparse
import os
import glob
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from PIL import Image
import h5py

from models.eig.networks.network import EIG
from models.eig.networks.network_classifier import EIG_classifier
from models.id.networks.network import VGG

from utils import config

CONFIG = config.Config()

def load_image(image, size):
    image = image.resize(size)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = np.moveaxis(image, 2, 0)
    image = image.astype(np.float32)
    return image

models_d = {

    'eig' : EIG(),

}

image_sizes = {

    'eig' : (227, 227),

}


def main():
    parser = argparse.ArgumentParser(description='Predictions of the models on the neural image test sets')
    parser.add_argument('--imagefolder',  type=str, default='./demo_images/',
                        help='Folder containing the input images.')
    parser.add_argument('--segment', help='whether to initially perform segmentation on the input images.',
                       action='store_true')
    parser.add_argument('--addoffset', help='whether to add offset away from the image boundary to the output of the segmentation step.',
                       action='store_true')
    parser.add_argument('--resume', type = str, default='', 
                        help='Where is the model weights stored if other than where the configuration file specifies.')

    global args
    args = parser.parse_args()

    print("=> Construct the model...")
    model = models_d['eig']
    model.cuda()

    resume_path = args.resume
    if resume_path == '':
        resume_path = os.path.join(CONFIG['PATHS', 'checkpoints'], 'eig', 'checkpoint_bfm.pth.tar')
    checkpoint = torch.load(resume_path)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume_path, checkpoint['epoch']))

    # test
    if not os.path.exists('./output'):
        os.mkdir('./output')
    outfile = os.path.join('./output', 'infer_output.hdf5')

    test(model, outfile)

def test(model, outfile):

    dtype = torch.FloatTensor

    path = args.imagefolder

    filenames = sorted(glob.glob(os.path.join(path, '*.png')))
    N = len(filenames)


    latents = []
    attended = []
    for i in range(N):
        fname = filenames[i]

        v = Image.open(fname)
        image = load_image(v, image_sizes['eig'])
        image = torch.from_numpy(image).type(dtype).cuda()
        image = image.unsqueeze(0)

        att, _, _, latent = model(image, segment=args.segment, add_offset=args.addoffset and args.segment, test=True)

        latents.append(latent.detach()[0].cpu().numpy().flatten())
        attended.append(att.detach()[0].cpu().numpy().flatten())

    f = h5py.File(outfile, 'w')
    f.create_dataset('number_of_layers', data=np.array([2]))
    f.create_dataset('latents', data=np.array(latents))
    f.create_dataset('Att', data=np.array(attended))

    asciiList = [n.split('/')[2].encode("ascii", "ignore") for n in filenames]
    f.create_dataset('filenames', (len(asciiList), 1), 'S10', data=asciiList)
    f.close()


if __name__ == '__main__':
    main()


