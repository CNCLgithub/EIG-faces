import io
import h5py
import copy
import numpy as np
from PIL import Image

from datasets import abstract


#-----------------------------------------------------------------#
#                        Dataset Class                            #
#-----------------------------------------------------------------#

class Background(abstract.HDF5Dataset):

    '''
    Dataset class for BFM09.
    '''

    def __init__(self, source, root = "/", input_shape = (3,227,227),
                 norms=False, no_check = True, trial_range=tuple()):

        self.input_shape = input_shape
        self.norms = norms
        self.no_check = no_check
        self.trial_range = trial_range
        super(Background, self).__init__(source, root)

    #-----------------------------------------------------------------#
    #                       For DatasetMixin                          #
    #-----------------------------------------------------------------#
    def __len__(self):
        return len(self._trials)

    @property
    def size(self):
        return len(self._trials)

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
        img = self._trials[i]
        parts = copy.copy(self.parts_template)
        parts['image'] = [img]
        return parts


    #-----------------------------------------------------------------#
    #                         Initialization                          #
    #-----------------------------------------------------------------#
    def find_trials(self, root):
        '''
        Looks for all valid trials available.
        Input:
            - root (h5py.Group): Group containing each trial of the form:
                `root/
                    0.png
                    1.png
                ...`
        '''
        trials = []
        # print('search for trials')
        n_trials = len(root)
        pot_trials = list(root.keys())

        if self.no_check:
            print("NOT checking trials. Maybe encounter errors "+\
                  "during iteration")
            trials = np.array(pot_trials)

        else:
            for trial in pot_trials:
                img = '{0!s}.png'.format(trial)
                if not img in root:
                    print("WARNING: Trial {} has no image...skipping".format(
                    trial))
                    continue

                trials.append(scene)

        self._trials = trials


    def find_means(self, root):
        '''
        Searches for the mean under the `root` path.
        Input:
            - root (h5py.Group): Group containing the following datasets:
                    1) `input_mean`
                    2) `input_sigma`
                    3) `target_max`
        '''
        self.input_mean = np.zeros(self.input_shape)
        self.input_sigma = np.ones(self.input_shape)

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
        return image

    #-----------------------------------------------------------------#
    #                           Data specific                         #
    #-----------------------------------------------------------------#
    def get_image(self, image):
        s = image.tostring()
        f = io.BytesIO(s)
        image = Image.open(f)
        image = image.resize(self.input_shape[1:])
        return image
