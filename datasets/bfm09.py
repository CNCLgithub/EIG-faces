import io
import h5py
import copy
import numpy as np
from PIL import Image, ImageOps

from datasets import BFM


#-----------------------------------------------------------------#
#                        Dataset Class                            #
#-----------------------------------------------------------------#

class BFM09(BFM):

    '''
    Dataset class for BFM09.
    '''

    def __init__(self,
                 source,
                 target_shape = (404,),
                 vgg = False,
                 **kw
    ):

        self.target_shape = target_shape
        self.vgg = vgg
        super(BFM09, self).__init__(source, **kw)


    #-----------------------------------------------------------------#
    #                       For DatasetMixin                          #
    #-----------------------------------------------------------------#
    def __len__(self):
        return self.size

    @property
    def size(self):
        return self._size


    def trials(self, i, f):
        '''
        Returns a dictionary of paths per part in the trial
        Args:
            i (int): the trial index
            f (file handle): root of the dataset
        Returns:
            parts (dict) : paths to data organized with keys corresponding to
                `self.trail_funcs`.
        '''
        scene = '{0:d}'.format(i+1)
        img = scene + '.png'
        if self.vgg:
            params = 'coeffs/{0!s}_vgg.txt'.format(scene)
        else:
            params = 'coeffs/{0!s}.txt'.format(scene)

        parts = copy.copy(self.parts_template)
        parts['image'] = [img]
        parts['params'] = [params]

        parts['flip'] = self.augment and np.random.sample() < 0.5
        parts['gray'] = self.augment and np.random.sample() < 0.1 #0.05

        return parts

    def process_trial(self, parts):
        '''
        Configures the parts of a trial into the final (input, target) tuple.
        '''
        image = parts['image'][0]
        target = parts['params'][0]

        if parts['flip']:
            if self.raw_image:
                image = ImageOps.mirror(image)
            else:
                image = np.flip(image, 1)

            target[400] *= -1
            target[402] *= -1

        if parts['gray']:
            if self.raw_image:
                image = image.convert('L').convert('RGB')
            else:
                raise RunTimeError("NotImplemented")

        image = np.asarray(image)[:,:,0:3]
        image = np.moveaxis(image, 2, 0)

        if not self.raw_image:
            image = np.around((image - self.input_mean) / self.input_sigma, 8)

        return image, target
