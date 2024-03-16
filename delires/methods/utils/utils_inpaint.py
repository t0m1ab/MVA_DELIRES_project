# -*- coding: utf-8 -*-
import numpy as np
import torch
import delires.methods.diffpir.utils.utils_image as util

### Mask generator from https://github.com/DPS2022/diffusion-posterior-sampling/

def random_sq_bbox(mask_shape, img_shape=(256, 256), n_channels=3, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    H, W = img_shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = H - margin_height - h
    maxl = W - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = np.ones((H, W, n_channels))
    mask[t:t+h, l:l+w, :] = 0

    return mask.astype(bool), t, t+h, l, l+w

class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 img_shape=(256, 256), n_channels=3, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.img_shape = img_shape
        self.n_channels = n_channels
        self.margin = margin

    def _retrieve_box(self):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(mask_shape=(mask_h, mask_w),
                              n_channels = self.n_channels, img_shape=self.img_shape,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self):
        H, W = self.img_shape
        total = H*W
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = np.ones(total)
        samples = np.random.choice(total, int(total * prob), replace=False)
        mask_vec[samples] = 0
        mask = mask_vec.reshape((H, W, 1))
        mask = mask.repeat(self.n_channels, axis=2)
        return mask.astype(bool)

    def __call__(self):
        if self.mask_type == 'random':
            mask = self._retrieve_random()
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box()
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box()
            mask = 1. - mask
            return mask

    


