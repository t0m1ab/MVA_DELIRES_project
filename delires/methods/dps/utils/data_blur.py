import torch
import torch.utils.data as data
from scipy.io import loadmat
import random

import utils.utils_agem as agem
import utils.utils_image as util


class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.opt = opt
        self.n_channels = 3
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.kernels = loadmat(opt['dataroot_kernels'])['kernels'][0]
        self.ksize = (33,33)

        assert self.paths_H, 'Error: H paths are empty.'

    def __getitem__(self, index):
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2tensor4(img_H)

        kernel = random.choice(self.kernels)
        kernel = torch.FloatTensor(kernel)[None, None]
        kernel = agem.pad_kernel(kernel, self.ksize)

        sigma = random.choice([(5/255), (10/255), (20/255)])

        img_L = agem.fft_blur(img_H, kernel)
        img_L = img_L + torch.randn(img_L.size()) * sigma

        return {'H':img_H, 'L':img_L , 'sigma': sigma, 'kernel': kernel}

    def __len__(self):
        return len(self.paths_H)
