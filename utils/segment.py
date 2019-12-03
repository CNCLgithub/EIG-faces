import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from PIL import Image, ImageOps

import datasets

from models.eig.networks.network import EIG

from utils import config

CONFIG = config.Config()

def main():
    parser = argparse.ArgumentParser(description='Segmentation using the VRN network')
    parser.add_argument('out', type = str,
        help='Directory to output the result')
    parser.add_argument('--dataset', type = str, help = 'Path to dataset',
                        default = 'bfm09_backface_culling.hdf5')
    parser.add_argument('--background', type = str, help = 'Path to background',
                        default = 'dtd_all.hdf5')
    parser.add_argument("--image", "-is", type = int, nargs="+",
                        default = (3,227,227), help = "Size of images. Default: 256x256x3")

    global args
    args = parser.parse_args()
    # create model using the pretrained alexnet.
    print("=> Construct the model...")
    
    model = EIG()
    model.cuda()

    if not os.path.exists(args.out):
        os.mkdir(args.out)
    if not os.path.exists(os.path.join(args.out, 'coeffs')):
        os.mkdir(os.path.join(args.out, 'coeffs'))

    print("Output location: {}".format(args.out))

    # Initialize both the foreground and background datasets using the background class
    d = datasets.BFM09(os.path.join(CONFIG['PATHS', 'databases'], args.dataset), raw_image = True, input_shape = args.image, augment = False)
    b = datasets.Background(os.path.join(CONFIG['PATHS', 'databases'], args.background), input_shape = args.image)
    train_loader = datasets.BFMOverlay(d,b)
    
    segment(train_loader, model)

def segment(train_loader, model):
    dtype = torch.FloatTensor

    # switch to train mode
    model.train()

    # shuffle trials
    N = len(train_loader)
    indices = range(N)

    BATCHSIZE = 1

    batches = np.array_split(indices, N/BATCHSIZE)

    for batch in batches:
        print('At ' + str(batch[0]))

        if os.path.exists(args.out + str(batch[0]+1) + ".png"):
            continue

        # load batch
        with train_loader.withbackground(True):
            trials = train_loader[batch]
        # if you want to run the batch through the network
        # instead of going one trial at a time, you can do:
        images, targets = zip(*trials)
        images = np.array(images)

        input_var = torch.from_numpy(images).type(dtype).cuda()

        segmented, _, _, _ = model(input_var, segment=True)
        
        segmented = segmented[0].detach().cpu().numpy()
        segmented = np.moveaxis(segmented, 0, 2)
        segmented = segmented.astype('uint8')
        segmented = Image.fromarray(segmented)
        segmented.save(os.path.join(args.out, str(batch[0]+1) + '.png'))

        # write targets
        np.savetxt(os.path.join(args.out, 'coeffs', str(batch[0]+1)+'.txt'), np.array(targets[0]))

if __name__ == '__main__':
    main()

