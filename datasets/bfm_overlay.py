import io
import h5py
import copy
import numpy as np
from PIL import Image, ImageOps
from contextlib import contextmanager

import datasets


class BFMOverlay:

    """
    Overlays a randomly sampled background over images from a
    `datasets.BFM09` dataset.

    For optimal IO interactions, datasets should be accessed
    in batches (`list` or `slice`). Every access using `__getitem__`
    requires a file object.
    """

    def __init__(self, dataset, backgrounds, vae=False, scale=False):

        self.dataset = dataset
        self.backgrounds = backgrounds
        self.no_background = False
        self.vae = vae
        self.scale = scale

    @property
    def dataset(self):
        """
        Access to the `datasets.BFM09` instance
        """
        return self._dataset

    @dataset.setter
    def dataset(self, d):
        if not isinstance(d, datasets.BFM09):
            raise ValueError('Unknown dataset type')
        if not d.raw_image:
            raise ValueError('Dataset.raw_image must be `True`.')
        self._dataset = d

    @property
    def backgrounds(self):
        """
        Access to the `datasets.Background` instance.
        """
        return self._backgrounds

    @backgrounds.setter
    def backgrounds(self, b):
        if not isinstance(b, datasets.Background):
            raise ValueError('Unknown background type')
        b.input_mean = self.dataset.input_mean
        b.input_sigma = self.dataset.input_sigma
        self._backgrounds = b


    def __len__(self):
        return len(self.dataset)

    @property
    def size(self):
        return len(self)

    @contextmanager
    def withbackground(self, b = True):
        """
        Context management for adding background image.
        """
        old_b = self.no_background
        self.no_background = not b
        yield
        self.no_background = old_b

    def __getitem__(self, idx):
        """
        Retrieves the (image,targets) for a slice/index and overlays those
        images onto randomly selected backgrounds.
        """
        if isinstance(idx, int):
            batch = [self.dataset[idx]]
        else:
            batch = self.dataset[idx]

        images, targets = zip(*batch)

        if self.no_background:

            images = [process_img(i) for i in images]

        else:
            n_trials = len(images)
            n_back = len(self.backgrounds)
            back_inds = np.random.choice(n_back, n_trials)
            backgrounds = self.backgrounds[back_inds]

            images = [self.overlay(back, img)
                      for back,img in zip(backgrounds, images)]

        if self.vae:
            targets = [resize(i, (64,64)) for i in images]

        pairs = list(zip(images, targets))
        if isinstance(idx, int):
            pairs = pairs[0]
        return pairs

    def overlay(self, background, foreground):
        """
        Overlays the forground image onto the background.
        """
        bg = copy.deepcopy(background)
        fg = copy.deepcopy(foreground)

        if self.scale == True:
            fg = self.random_resize(fg)

        fg_bw = fg.convert(mode = '1')

        inds = np.mean(np.asarray(fg), axis = 2)
        mask = np.zeros(inds.shape)
        inds = np.where(inds > 250)
        # mask[0:25, 0:25] = 255
        mask[inds] = 255
        mask = Image.fromarray(mask) # , mode = 'L')
        mask = mask.convert(mode = 'L')
        img = Image.composite(bg, fg, mask)
        img = process_img(img)
        return img


    def random_resize(self, img):
        """
        Randomly scales the foreground image.
        """
        im_size = img.size[1]
        new_size = int(im_size * (np.random.sample() * 0.25 + 1.0))
        
        new_fg = Image.new('RGB',
                           (new_size, new_size),
                           (255, 255, 255))
        
        translate_x = int(np.random.sample() * (new_size - im_size))
        translate_y = int(np.random.sample() * (new_size - im_size))

        new_fg.paste(img, (translate_x, translate_y))

        new_fg = new_fg.resize((im_size, im_size))

        return new_fg

def process_img(img):
    """
    Formats an `Image` to `np.ndarray`. 
    """
    img = np.asarray(img)[:,:,0:3]
    img = np.moveaxis(img, 2, 0)
    return img.astype(np.float32)

def resize(img, size):
    """
    Resizes an array format image.
    """
    img = np.moveaxis(img, 0, 2).astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize(size)
    return process_img(img)
