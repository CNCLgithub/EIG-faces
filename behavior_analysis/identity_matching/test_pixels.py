import argparse
import os
import shutil
import time
import h5py

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from PIL import Image

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

def main():
    '''
    Test pixel similarity as a basic image matching strategy across the identity matching tasks.
    '''

    if not os.path.exists('./output'):
        os.mkdir('./output')
    outfile = './output/image_matching_pixels.hdf5'

    # test
    test(outfile)

def test(outfile):
    path = os.path.join(CONFIG['PATHS', 'behavior'], 'stimuli/identity_matching')

    N = 96

    f = h5py.File(outfile, 'w')
    for counter, exp in enumerate(['/exp1/', '/exp2/', '/exp3/']):
        z = []

        for i in range(1, N+1):
            for j in range(2):
                fname = path + exp + str(i) + '_' + str(j+1) + '.png'

                v = Image.open(fname)
                image = load_image(v, (227, 227))
                z.append(image.flatten())

        f.create_dataset(str(counter), data=np.array(z))

    f.close()


if __name__ == '__main__':
    main()


