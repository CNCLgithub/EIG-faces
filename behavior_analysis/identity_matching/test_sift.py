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

import cv2

from utils import config
CONFIG = config.Config()


def main():
    '''
    Compute and test SIFT as a basic image matching strategy across the identity matching tasks. The results of this script are pre-computed and shared with the code repo.
    '''

    if not os.path.exists('./output'):
        os.mkdir('./output')
    outfile = './output/image_matching_sift.hdf5'

    # test
    test(outfile)

def test(outfile):
    path = os.path.join(CONFIG['PATHS', 'behavior'], 'stimuli/identity_matching')

    N = 96

    f = h5py.File(outfile, 'w')
    sift = cv2.xfeatures2d.SIFT_create()
    for counter, exp in enumerate(['/exp1/', '/exp2/', '/exp3/']):
        z = []

        for i in range(1, N+1):
            for j in range(2):
                fname = path + exp + str(i) + '_' + str(j+1) + '.png'

                image = cv2.imread(fname)
                image = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
                kp, desc = sift.detectAndCompute(image, None)
                z.append(desc[0].flatten())

        f.create_dataset(str(counter), data=np.array(z))

    f.close()


if __name__ == '__main__':
    main()


