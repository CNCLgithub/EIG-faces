import sys
import h5py
import numpy as np
from abc import ABC, abstractmethod


#-----------------------------------------------------------------#
#                       Abstract Class                            #
#-----------------------------------------------------------------#
'''
HDF5Dataset:
- A abstract class for hdf5 dataset interfacing
public methods:
    __len__ : Returns the number of trials found
    get_example : Given an integer, returns a tuple containting the input
        and the ground truth for that trial
private methods:
    find_means : Defines where the means should be present and how to process
        those means for regularization
    init_trial_func : Defines the set of functions to process the different
        components (images, labels, audio, ..etc) of each trial. Also, this
        function must assign `self.process_trial` which performs an global
        processing on the trial needed prior to training.
abstract methods:
    __len__
    find_trials :: file object -> list of dataset paths
    find_means :: file object -> numpy array
    init_trial_func :: -> [functions], process_trial
'''
class HDF5Dataset(ABC):

    def __init__(self, source, root='/', debug = False):

        self.source = source
        self.root = root
        self.debug = debug
        self.trial_funcs = []
        self._initialize()
        self._check()

    #-----------------------------------------------------------------#
    #                       For DatasetMixin                          #
    #-----------------------------------------------------------------#
    @abstractmethod
    def __len__(self):
        pass

    # int -> trial
    def get_example(self, index, f = None):
        """
        Returns the trial corresponding to the given index.
        """
        trial_parts = self.get_trial(index, f)
        ex = self.process_trial(trial_parts)
        return ex

    def __getitem__(self, idx):

        with h5py.File(self.source, 'r') as f:
            root = f[self.root]
            if isinstance(idx, slice):
                return [self.get_example(ii, root)
                    for ii in range(*idx.indices(len(self)))]
            elif isinstance(idx, list) or isinstance(idx, np.ndarray):
                return [self.get_example(i, root) for i in idx]
            else:
                if idx > len(self) - 1:
                    raise IndexError("Requested trial {0:d} for dataset of length {1:d}".format(
                    idx, len(self)))
                elif idx < 0:
                    raise IndexError("Requested trial with negative index")
                else:
                    return self.get_example(idx, root)
    #-----------------------------------------------------------------#
    #                         Initialization                          #
    #-----------------------------------------------------------------#
    def _initialize(self):

        with h5py.File(self.source, 'r') as f:
            print("searching for dataset means...")
            self.find_means(f)
            print("searching for dataset means... Done")

            print("determining dataset size...")
            self.find_trials(f[self.root])
            print("determining dataset size... Done")
            print("Dataset has {} trials".format(self.size))

        print("initializing data retrieval...")
        self.init_trial_func()
        print("initializing data retrieval... Done")

        if self.debug:
            size = min(self.size, 10)
            print("RUNNING IN DEBUG MODE. WILL ONLY HAVE SIZE OF {}".format(size))
            self.size = size


    def _check(self):

        if len(self.trial_funcs) == 0:
            raise NotImplementedError("Must have trial functions")


    @abstractmethod
    def find_trials(self, f):
        pass

    @abstractmethod
    def init_trial_func(self):
        pass

    @abstractmethod
    def process_trial(self, parts):
        pass

    #-----------------------------------------------------------------#
    #                         General Helpers                         #
    #-----------------------------------------------------------------#

    def get_trial(self, i, r):

        trial_parts  = self.trials(i, r)
        trial_funcs = self.trial_funcs
        parts = {}

        for key in trial_parts:
            if key in trial_funcs:
                paths = trial_parts[key]
                p_func = trial_funcs[key]
                part = []
                for path in paths:
                    try:
                        raw = r[path].value
                    except:
                        msg = '{} not found'.format(path)
                        raise KeyError(msg)
                    v = p_func(raw)
                    part.append(v)
            else:
                part = trial_parts[key]

            parts[key] = part

        return parts
