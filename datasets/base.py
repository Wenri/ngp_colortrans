from collections import namedtuple
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    IMAGE_SIZE = namedtuple('IMAGE_SIZE', ['w', 'h'])

    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.batch_size = 1
        self.ray_sampling_strategy = 'all_images'
        self.poses = None
        self.rays = None
        self.img_wh = None

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def _getitem_random(self):
        # training pose is retrieved in train.py
        match self.ray_sampling_strategy:
            case 'all_images':  # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            case 'same_image' | 'deferred_images':  # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            case _:
                raise ValueError(f"Unknown ray sampling strategy {self.ray_sampling_strategy}!")

        # randomly select pixels
        pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1], self.batch_size)
        rays = self.rays[img_idxs, pix_idxs]
        sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                  'rgb': rays[:, :3]}
        if self.rays.shape[-1] == 4:  # HDR-NeRF data
            sample['exposure'] = rays[:, 3:]
        if self.ray_sampling_strategy.startswith('deferred'):
            sample['img'] = rearrange(self.rays[img_idxs, :, :3], '... (h w) c -> ... h w c',
                                      w=self.img_wh[0], h=self.img_wh[1])
        return sample

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            return self._getitem_random()

        sample = {'pose': self.poses[idx], 'img_idxs': idx}
        if len(self.rays) > 0:  # if ground truth available
            rays = self.rays[idx]
            sample['rgb'] = rays[:, :3]
            if rays.shape[1] == 4:  # HDR-NeRF data
                sample['exposure'] = rays[0, 3]  # same exposure for all rays

        return sample
