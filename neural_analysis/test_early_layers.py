'''
A hacky way to extract features in the early convoutional layers of the models.
'''

import argparse
import os
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

from models.eig.networks.network_classifier_early_layers import EIG_classifier_early_layers
#from models.id.networks.network_early_layers import VGG_early_layers

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

    'eig_classifier' : lambda x: EIG_classifier_early_layers(x),
    'vgg' : lambda x: VGG_early_layers(x),

}

image_sizes = {

    'eig_classifier' : (227, 227),
    'vgg' : (224, 224),

}

def main():
    parser = argparse.ArgumentParser(description='Predictions of the models on the neural image test sets')
    parser.add_argument('model', type=str, help = 'Which model is being tested: eig_classifier, vgg, vgg_raw')
    parser.add_argument('imageset',  type=str,
                        help='Test with BFM (bfm) images or FIV (fiv) images?')
    parser.add_argument('--segment', help='whether to initially perform segmentation on the input images.',
                       action='store_true')
    parser.add_argument('--resume', type = str, default='', 
                        help='Where is the model weights stored if other than where the configuration file specifies.')

    global args
    args = parser.parse_args()

    assert args.imageset in ['bfm', 'fiv'], 'set imageset to either bfm or fiv; e.g., --imageset fiv'

    print("=> Construct the model...")
    model = models_d[args.model](args.imageset)
    model.cuda()
    print('Loaded network weights.')

    # test
    if not os.path.exists('./output'):
        os.mkdir('./output')
    outfile = os.path.join('./output', args.model + '_' + args.imageset + '_early_layers.hdf5')
    test(model, outfile)

def test(model, outfile):

    path = os.path.join(CONFIG['PATHS', 'neural'], 'stimuli', args.imageset)

    N = 175
    dtype = torch.FloatTensor

    if 'eig' not in args.model:
        model.eval()

    all_layers = {}
    for j in range(4):
        all_layers[j] = []

    for i in range(1, N+1):
        fname = os.path.join(path, str(i) + '.png')

        v = Image.open(fname)
        image = load_image(v, image_sizes[args.model])
        image = torch.from_numpy(image).type(dtype).cuda()
        image = image.unsqueeze(0)

        if 'eig' in args.model:
            outputs = model(image, segment=args.segment, add_offset=args.imageset=='fiv')
        else:
            outputs = model(image, test=True)

        for j in range(len(outputs)):
            all_layers[j].append(outputs[j])

    f = h5py.File(outfile, 'w')
    f.create_dataset('number_of_layers', data=np.array([len(outputs)]))
    for j in range(len(outputs)):
        f.create_dataset(str(j), data=np.array(all_layers[j]))

    f.close()


if __name__ == '__main__':
    main()


