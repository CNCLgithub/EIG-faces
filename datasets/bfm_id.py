import io
import h5py
import copy
import numpy as np
from PIL import Image, ImageOps

from datasets import BFM09

#-----------------------------------------------------------------#
#                        Dataset Class                            #
#-----------------------------------------------------------------#

class BFMId(BFM09):

    '''
    Dataset class for BFMId, where targets are the IDs of each
    image.
    '''

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
        idx = i + 1
        scene = '{0:d}'.format(idx)
        img = scene + '.png'

        parts = copy.copy(self.parts_template)
        parts['image'] = [img]
        parts['flip'] = self.augment and np.random.sample() >= 0.5
        parts['gray'] = self.augment and np.random.sample() < 0.1 #0.05 
        parts['id'] = [np.floor(idx / 400)]

        return parts


    #-----------------------------------------------------------------#
    #                         Initialization                          #
    #-----------------------------------------------------------------#


    def find_means(self, root):
        '''
        Searches for the mean under the `root` path.
        Input:
            - root (h5py.Group): Group containing the following datasets:
                    1) `input_mean`
                    2) `input_sigma`
                    3) `target_max`
        '''
        if 'input' in self.norms:
            self.input_mean = root["input_mean"][()]
            self.input_sigma = root["input_sigma"][()]

        else:
            self.input_mean = np.zeros(self.input_shape)
            self.input_sigma = np.ones(self.input_shape)

        if 'target' in self.norms:
            self.target_max = root['target_max'][()]
        else:
            self.target_max = np.ones(1)


    def init_trial_func(self):
        '''
        Determines the format for data loading used in `process_trial`.
        '''
        trial_funcs = {	# inputs
                        'image' : self.get_image,
                        }

        self.trial_funcs = trial_funcs
        self.parts_template = {k : [] for k in trial_funcs}


    def process_trial(self, parts):
        '''
        Configures the parts of a trial into the final (input, target) tuple.
        '''
        image = parts['image'][0]
        target = parts['id'][0]

        if parts['flip']:
            if self.raw_image:
                image = ImageOps.mirror(image)
            else:
                image = np.flip(image, 1)

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

