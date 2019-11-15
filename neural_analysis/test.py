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
    image /= normalize_d[args.model][0]
    image -= normalize_d[args.model][1]
    return image

models_d = {

    'eig' : EIG(),
    'eig_classifier' : EIG_classifier(),
    'vgg_fiv' : VGG(25),
    'vgg_bfm' : VGG(500),
    'vgg_raw' : VGG(),

}

image_sizes = {

    'eig' : (227, 227),
    'eig_classifier' : (227, 227),
    'vgg' : (224, 224),
    'vgg_raw' : (224, 224),

}

normalize_d = {

    'eig' : [1., 0],
    'eig_classifier' : [1., 0],
    'vgg' : [1., 0],
    'vgg_raw' : [1., 0],

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
    if args.model == 'vgg':
        if args.imageset == 'bfm':
            model = models_d['vgg_bfm']
        elif args.imageset == 'fiv':
            model = models_d['vgg_fiv']
    else:
        model = models_d[args.model]
    model.cuda()

    if args.model != 'vgg_raw':
        resume_path = args.resume
        if resume_path == '':
            resume_path = os.path.join(CONFIG['PATHS', 'checkpoints'], args.model, 'checkpoint_'+args.imageset+'.pth.tar')
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, checkpoint['epoch']))
    else:
        print('Loaded pre-trained network.')

    # test
    if not os.path.exists('./output'):
        os.mkdir('./output')
    outfile = os.path.join('./output', args.model + '_' + args.imageset + '.hdf5')
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
    attended = []
    for i in range(1, N+1):
        fname = os.path.join(path, str(i) + '.png')

        v = Image.open(fname)
        image = load_image(v, image_sizes[args.model])
        image = torch.from_numpy(image).type(dtype).cuda()
        image = image.unsqueeze(0)

        if 'eig' in args.model:
            att, out1, out2, out3 = model(image, segment=args.segment, add_offset=args.imageset=='fiv', test=True)
            outputs = [out1, out2, out3]
        else:
            outputs = model(image, test=True)

        for j in range(len(outputs)):
            all_layers[j].append(outputs[j].detach()[0].cpu().numpy().flatten())
        if 'eig' in args.model:
            attended.append(att.detach()[0].cpu().numpy().flatten())

    f = h5py.File(outfile, 'w')
    f.create_dataset('number_of_layers', data=np.array([len(outputs)]))
    for j in range(len(outputs)):
        f.create_dataset(str(j), data=np.array(all_layers[j]))
    if 'eig' in args.model:
        f.create_dataset('Att', data=np.array(attended))
    f.close()


if __name__ == '__main__':
    main()


