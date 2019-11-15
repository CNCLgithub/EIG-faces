import io
import h5py
import copy
import numpy as np
from PIL import Image, ImageOps
from abc import ABC, abstractmethod

from datasets.abstract import HDF5Dataset


#-----------------------------------------------------------------#
#                        Dataset Class                            #
#-----------------------------------------------------------------#

class BFM(HDF5Dataset, ABC):

    '''
    Basal-face model interfance handling high-level interactions for HDF5.
    Should never be initialized directly.
    '''

    def __init__(self,
                 source,
                 input_shape = (3, 227 , 227),
                 norms=[],
                 no_check = True,
                 trial_range=tuple(),
                 raw_image = False,
                 augment = True,
                 **kw
    ):

        self.input_shape = input_shape
        self.norms = norms
        self.no_check = no_check
        self.trial_range = trial_range
        self.raw_image = raw_image
        self.augment = augment
        super(BFM, self).__init__(source, **kw)


    #-----------------------------------------------------------------#
    #                       For DatasetMixin                          #
    #-----------------------------------------------------------------#
    def __len__(self):
        return self.size

    @property
    def size(self):
        return self._size



    #-----------------------------------------------------------------#
    #                         Initialization                          #
    #-----------------------------------------------------------------#
    def find_trials(self, root):
        '''
        Looks for all valid trials available.
        Assumes 200k trials.

        Input:
            - root (h5py.Group): Group containing each trial of the form:
                `root/
                    coef/
                        i[_vgg].txt
                        ....
                    0.png
                    1.png
                ...`
        '''
        trials = []
        pot_trials = np.arange(1, 200001)
        if len(self.trial_range) > 0:
            pot_trials = pot_trials[self.trial_range[0]:self.trial_range[1]]


        if self.no_check:
            print("NOT checking trials. Maybe encounter errors "+\
                  "during iteration")
            trials = np.array(pot_trials)

        else:
            msg = 'BFM does not currently support checking'
            raise NotImplementedError(msg)
            # for trial in pot_trials:
            #     img = '{0:d}.png'.format(trial)
            #     params = 'coeff/{0:d}_vgg.txt'.format(trial)
            #     if not img in root:
            #         print("WARNING: Trial {} has no image...skipping".format(
            #         trial))
            #         continue
            #     if not params in root:
            #         print("WARNING: Trial {} has no parameters...skipping".format(
            #             trial))
            #         continue

            #     trials.append(scene)

        self._size = len(trials)


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
            self.input_mean = root["input_mean"].value
            self.input_sigma = root["input_sigma"].value

        else:
            self.input_mean = np.zeros(self.input_shape)
            self.input_sigma = np.ones(self.input_shape)

        if 'target' in self.norms:
            self.target_max = root['target_max'].value
        else:
            self.target_max = np.ones(self.target_shape)


    def init_trial_func(self):
        '''
        Determines the format for data loading used in `process_trial`.
        '''
        trial_funcs = {	# inputs
                        'image' : self.get_image,
                        # targets
                        'params' : self.get_param
                        }

        self.trial_funcs = trial_funcs
        self.parts_template = {k : [] for k in trial_funcs}

    @abstractmethod
    def process_trial(self, parts):
        '''
        Configures the parts of a trial into the final (input, target) tuple.
        '''
        pass

    @abstractmethod
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
        pass

    #-----------------------------------------------------------------#
    #                           Data specific                         #
    #-----------------------------------------------------------------#
    def get_image(self, image):
        """
        Either returns a raw image using `Image` or a 3xSIZE np.array.
        """
        s = image.tostring()
        f = io.BytesIO(s)
        image = Image.open(f)
        image = image.resize(self.input_shape[1:])

        image = image.convert('RGB')

        if self.raw_image:
            return image

        image = np.asarray(image)[:,:,0:3]
        image = np.moveaxis(image, 2, 0)
        image = image.astype(np.float32)
        return image.astype(np.float32)

    def get_param(self, param):
        ba = bytearray(param)
        s = ba.decode('ascii')
        if '\n' in s[:-1]:
            return np.fromstring(s, sep='\n')
        else:
            return np.fromstring(s, sep=',')
            

    '''
    Helper that loads numpy byte files
    '''
    def get_array(self, bs):
        s = bs.tostring()
        f = io.BytesIO(s)
        v = np.load(f)
        return v
