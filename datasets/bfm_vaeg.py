import copy
import inspect
import numpy as np

from datasets import BFM09

#-----------------------------------------------------------------#
#                        Dataset Class                            #
#-----------------------------------------------------------------#

class BFM_VAEG(BFM09):

    parents = {
        '09': BFM09,
    }
    #-----------------------------------------------------------------#
    #                         Initialization                          #
    #-----------------------------------------------------------------#

    def __init__(self, source, mode, batchsize = 20, flip = False, **kw):

        self.mode = mode
        self.batchsize = 20
        kw['flip'] = flip
        self.super.__init__(self, source, **kw)
        # self.super.__init__(source, **kw)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, m):
        if not m in ['09', '2017']:
            raise ValueError('Mode must be either \"09\" or \"2017\"')
        if hasattr(self, '_mode'):
            raise ValueError('Property `mode` is immutable')

        self._mode = m
        f_super = lambda c: (lambda s : super(c, s))

        print('Loading with setting: {}'.format(m))
        self._super = self.parents[m]

        # if m == '09':
        #     # self._super = f_super(BFM2017)
        #     self._super = # f_super(BFM09)
        # else:
        #     self._super = # f_super(BFM2017)

    @property
    def super(self):
        """
        Emulates change in MRO based on using `BFM09` and `BFM2017`.
        """
        return self._super

    @property
    def size(self):
        return self._size

    @property
    def batchsize(self):
        return self._k

    @batchsize.setter
    def batchsize(self, k):
        if not (k % 2 == 0):
            raise ValueError('Batchsize must be even')
        self._k = k

    def find_trials(self, root):
        '''
        Looks for all valid trials available.
        Input:
            - root (h5py.Group): Group containing each trial of the form:
                `root/
                    i (batch)/
                        0.png
                        0.csv
                        ....
                ...`
        '''
        trials = []
        pot_trials = np.arange(len(root.keys()))
        if len(self.trial_range) > 0:
            pot_trials = pot_trials[self.trial_range[0]:self.trial_range[1]]


        if self.no_check:
            print("NOT checking trials. Maybe encounter errors "+\
                  "during iteration")
            trials = np.array(pot_trials)

        else:
            msg = 'BFMVAEG does not currently support checking'
            raise NotImplementedError(msg)
        self._size = len(trials)

    def trials(self, i, f):
        '''
        For a given minibatch, randomly chooses `k` pairs of images along with
        the camera paremeters for each respective input.

        Args:
            i (int): the trial index
            f (file handle): root of the dataset
        Returns:
            parts (dict) : paths to data organized with keys corresponding to
                `self.trail_funcs`.
        '''
        batch = '{0:d}'.format(i)
        k = int(self.batchsize / 2)
        # randomly choose k pairs, first k are inputs
        idxs = np.random.permutation(k*2)
        images = [batch + '/{0:d}.png'.format(idx) for idx in idxs]
        params = [batch + '/{0:d}.csv'.format(idx) for idx in idxs]

        parts = copy.copy(self.parts_template)
        parts['image'] = images
        parts['params'] = params
        parts['flip'] = self.flip and np.random.sample() < 0.5
        return parts

    def process_trial(self, parts):
        '''
        Configures the parts of a trial into the final (input, target) tuple.
        '''
        k = int(self.batchsize / 2)
        enc_imgs = np.array(parts['image'][:k])
        enc_pose = np.array(parts['params'][:k])
        if parts['flip']:
            enc_images = np.flip(enc_imgs, 2)
            enc_pose[:, 400] *= -1
            enc_pose[:, 402] *= -1
        dec_imgs = np.array(parts['image'][k:])
        dec_pose = np.array(parts['params'][k:])

        return enc_imgs, enc_pose, dec_imgs, dec_pose

    def get_param(self, param):
        # print(inspect.getmembers(self.super, predicate=inspect.ismethod))
        # print(self.super)
        return self.super.get_param(self, param)
