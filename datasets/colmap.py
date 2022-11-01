import glob
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from misc.imagewrap import _make_L_matrix
from .base import BaseDataset
from .colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
from .color_utils import read_image
from .ray_utils import get_ray_directions, center_poses, create_spheric_poses
from .seg_util import read_seg


def _calc_coeffs(from_points, to_points):
    L = _make_L_matrix(from_points)
    V = np.resize(to_points, (len(to_points) + 3, 2))
    V[-3:, :] = 0
    coeffs = np.linalg.lstsq(L, V, rcond=None)  # np.dot(np.linalg.pinv(L), V)
    return coeffs[0]


class ColmapDataset(BaseDataset):
    log = logging.getLogger(__name__)
    _EPSILON = torch.finfo(torch.double).eps

    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()
        try:
            d = np.load('assets/transimg.npz')
            from_points = d['from_points'] / 128.
            to_points = d['to_points'] / 128.
            ref_points = d['ref_points'] / 128.
            self.from_points = torch.from_numpy(from_points)
            self.to_points = torch.from_numpy(to_points)
            self.ref_points = torch.from_numpy(ref_points)
            self.flow = torch.from_numpy(d['flow'])
            self.coeffs = torch.from_numpy(_calc_coeffs(from_points, to_points))
            self.rev_coeffs = torch.from_numpy(_calc_coeffs(to_points, from_points))
        except Exception as e:
            self.log.exception('From/To points not found. Assuming no warp.')
            self.from_points = None
            self.coeffs = None

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def _U(self, x: torch.Tensor):
        return x * torch.where(x < self._EPSILON, 0, torch.log(x) / 2)

    def _calculate_f(self, coeffs, x, y):
        w = coeffs[:-3]
        a1, ax, ay = coeffs[-3:]
        # The following may use too much RAM:
        points = self.from_points
        distances = self._U(torch.square(points[:, 0] - x[..., None]) + torch.square(points[:, 1] - y[..., None]))
        distances = (w * distances).sum(axis=-1)
        return a1 + ax * x + ay * y + distances

    def _trans_ab(self, img, skip=True):
        a, b = torch.unbind(img[..., 1:], dim=1)
        if not skip:
            a, b = self._calculate_f(self.coeffs[:, 0], a, b), self._calculate_f(self.coeffs[:, 1], a, b)
        return torch.stack((a, b), dim=1)

    def _img_trans(self, img: torch.Tensor, scale=(255., 128., 128.)):
        if self.from_points is not None:
            img = torch.cat((img, self._trans_ab(img)), dim=1)
            scale = torch.as_tensor(scale + scale[1:], dtype=img.dtype, device=img.device)
        else:
            scale = torch.as_tensor(scale, dtype=img.dtype, device=img.device)
        return torch.clamp((img / scale).to(torch.float32), -1, 1)

    def _read_imgbuf(self, img_path):
        img = read_image(img_path, self.img_wh)
        img = torch.FloatTensor(self._img_trans(img))
        buf = [img]  # buffer for ray attributes: rgb, etc

        if 'HDR-NeRF' in self.root_dir:  # get exposure
            folder = self.root_dir.split('/')
            scene = folder[-1] if folder[-1] != '' else folder[-2]
            if scene in ['bathroom', 'bear', 'chair', 'desk']:
                e_dict = {e: 1 / 8 * 4 ** e for e in range(5)}
            elif scene in ['diningroom', 'dog']:
                e_dict = {e: 1 / 16 * 4 ** e for e in range(5)}
            elif scene in ['sofa']:
                e_dict = {0: 0.25, 1: 1, 2: 2, 3: 4, 4: 16}
            elif scene in ['sponza']:
                e_dict = {0: 0.5, 1: 2, 2: 4, 3: 8, 4: 32}
            elif scene in ['box']:
                e_dict = {0: 2 / 3, 1: 1 / 3, 2: 1 / 6, 3: 0.1, 4: 0.05}
            elif scene in ['computer']:
                e_dict = {0: 1 / 3, 1: 1 / 8, 2: 1 / 15, 3: 1 / 30, 4: 1 / 60}
            elif scene in ['flower']:
                e_dict = {0: 1 / 3, 1: 1 / 6, 2: 0.1, 3: 0.05, 4: 1 / 45}
            elif scene in ['luckycat']:
                e_dict = {0: 2, 1: 1, 2: 0.5, 3: 0.25, 4: 0.125}
            e = int(img_path.split('.')[0][-1])
            buf.append(e_dict[e] * torch.ones_like(img[:, :1]))

        return torch.cat(buf, 1)

    def _read_imgseg(self, img_path):
        seg = read_seg(img_path, self.img_wh)
        return seg

    def sort_sem(self):
        prob = torch.softmax(self.segs, dim=2, dtype=torch.float32)
        prob = torch.sum(prob, dim=(0, 1))
        ind = torch.argsort(prob, descending=True)
        return ind

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height * self.downsample)
        w = int(camdata[1].width * self.downsample)
        self.img_wh = self.IMAGE_SIZE(w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * self.downsample
            cx = camdata[1].params[1] * self.downsample
            cy = camdata[1].params[2] * self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.downsample
            fy = camdata[1].params[1] * self.downsample
            cx = camdata[1].params[2] * self.downsample
            cy = camdata[1].params[3] * self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        self.directions = get_ray_directions(h, w, self.K)

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        if '360_v2' in self.root_dir and self.downsample < 1:  # mipnerf360 data
            folder = f'images_{int(1 / self.downsample)}'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3]  # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d])  # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        if split == 'test_traj':  # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return

        if 'HDR-NeRF' in self.root_dir:  # HDR-NeRF data
            if 'syndata' in self.root_dir:  # synthetic
                # first 17 are test, last 18 are train
                self.unit_exposure_rgb = 0.73
                if split == 'train':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                              f'train/*[024].png')))
                    self.poses = np.repeat(self.poses[-18:], 3, 0)
                elif split == 'test':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                              f'test/*[13].png')))
                    self.poses = np.repeat(self.poses[:17], 2, 0)
                else:
                    raise ValueError(f"split {split} is invalid for HDR-NeRF!")
            else:  # real
                self.unit_exposure_rgb = 0.5
                # even numbers are train, odd numbers are test
                if split == 'train':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                              f'input_images/*0.jpg')))[::2]
                    img_paths += sorted(glob.glob(os.path.join(self.root_dir,
                                                               f'input_images/*2.jpg')))[::2]
                    img_paths += sorted(glob.glob(os.path.join(self.root_dir,
                                                               f'input_images/*4.jpg')))[::2]
                    self.poses = np.tile(self.poses[::2], (3, 1, 1))
                elif split == 'test':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                              f'input_images/*1.jpg')))[1::2]
                    img_paths += sorted(glob.glob(os.path.join(self.root_dir,
                                                               f'input_images/*3.jpg')))[1::2]
                    self.poses = np.tile(self.poses[1::2], (2, 1, 1))
                else:
                    raise ValueError(f"split {split} is invalid for HDR-NeRF!")
        else:
            # use every 8th image as test set
            if split == 'train':
                img_paths = [x for i, x in enumerate(img_paths) if i % 8 != 0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i % 8 != 0])
            elif split == 'test':
                img_paths = [x for i, x in enumerate(img_paths) if i % 8 == 0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i % 8 == 0])

        print(f'Loading {len(img_paths)} {split} segs ...')
        self.segs = torch.stack(tuple(map(self._read_imgseg, tqdm(img_paths))))  # (N_images, hw, ?)

        print(f'Loading {len(img_paths)} {split} images ...')
        self.rays = torch.stack(tuple(map(self._read_imgbuf, tqdm(img_paths))))  # (N_images, hw, ?)

        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)
