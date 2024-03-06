import torch.utils.data as data
import utils.utils_image as util
import random


class DatasetH(data.Dataset):
    def __init__(self, config):
        super(DatasetH, self).__init__()
        self.config = config
        self.paths_H = util.get_image_paths(config.dataroot)
        self.patch_size = config.image_size
        assert self.paths_H, 'Error: H paths are empty.'

    def __getitem__(self, index):
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, 3)

        H, W = img_H.shape[:2]

        # ---------------------------------
        # randomly crop the patch
        # ---------------------------------
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))
        patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

        # ---------------------------------
        # augmentation - flip, rotate
        # ---------------------------------
        #mode = random.randint(0, 7)
        #patch_H = util.augment_img(patch_H, mode=mode)

        img_H = patch_H

        # ------------------------------------
        # HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H = util.uint2tensor3(img_H)
        
        img_H = (img_H - 0.5) * 2

        return {'H': img_H, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
