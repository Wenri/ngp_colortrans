from typing import Optional

import torch
from PIL import ImageCms
from PIL import Image
from kornia.color import lab_to_rgb
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES, N_CH

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss, HistLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings

warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams, palette=None):
        super().__init__()
        self.val_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/init'
        os.makedirs(self.val_dir, exist_ok=True)
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else None
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act)
        # self.model = NeRF(scale=self.hparams.scale, rgb_act=rgb_act)

        self.loss = NeRFLoss(self.model._N_COLOR_CH,
                             lambda_distortion=self.hparams.distortion_loss_w)
        self.deferred_loss = HistLoss()

        self.CLASSES = N_CH - self.model._N_COLOR_CH

        if palette is None:
            palette = np.random.randint(0, 255, size=(self.CLASSES, 3))
        self.palette = np.asarray(palette)
        assert palette.shape[0] == self.CLASSES
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

    def forward(self, batch, split):
        if split == 'train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split != 'train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1 / 256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        self.loss.setup_sem_ind(self.train_dataset.sort_sem())

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR', nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT', nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)]  # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr / 30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=0,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=0,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device), self.poses, self.train_dataset.img_wh)

    def deferred_step(self, img, pix_idxs, rays, b_save, **kwargs):
        w, h = self.train_dataset.img_wh
        kwargs['pose'] = self.poses[kwargs['img_idxs']]
        with torch.no_grad():
            results = self(kwargs, split='deferred')
        pred_img = results['rgb']

        is_finite = torch.all(torch.isfinite(pred_img))
        if b_save or not is_finite:
            self.save_image(pred_img, f'deferred_pred.png')
            self.save_image(rearrange(img, 'h w c -> (h w) c'), f'deferred_gt.png')
            assert is_finite

        pred_img.scatter_(dim=0, index=pix_idxs.unsqueeze(-1).expand_as(rays), src=rays)
        pred_img = rearrange(pred_img, '(h w) c -> 1 h w c', h=h)

        loss = self.deferred_loss(results=pred_img, target=rearrange(img, 'h w c -> 1 h w c'))

        return loss

    def training_step(self, batch, batch_nb, *args):
        if self.global_step % self.update_interval == 0:
            self.model.update_density_grid(0.01 * MAX_SAMPLES / 3 ** 0.5,
                                           self.global_step < self.warmup_steps,
                                           erode=False)  # self.hparams.dataset_name == 'colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results=results, target=batch)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                                               **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5 * (unit_exposure_rgb - self.train_dataset.unit_exposure_rgb) ** 2

        if 'img' in batch:
            loss_d['img'] = self.deferred_step(**batch, rays=results['rgb'], b_save=batch_nb == 0)

        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            scale = torch.as_tensor((100.0, 128.0, 128.0), dtype=batch['rgb'].dtype, device=batch['rgb'].device)
            rgb_pred = lab_to_rgb(rearrange(results['rgb'][..., :3] * scale, 'b c -> b c 1 1'))
            rgb_gt = lab_to_rgb(rearrange(batch['rgb'][..., :3] * scale, 'b c -> b c 1 1'))
            self.train_psnr(rgb_pred, rgb_gt)

        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples'] / len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples'] / len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/{self.current_epoch}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        w, h = self.train_dataset.img_wh
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        scale = torch.as_tensor((100.0, 128.0, 128.0), dtype=rgb_gt.dtype, device=rgb_gt.device)
        # compute each metric per image
        rgb_pred = lab_to_rgb(rearrange(results['rgb'][..., :3] * scale, '(h w) c -> 1 c h w', h=h))
        rgb_gt = lab_to_rgb(rearrange(rgb_gt[..., :3] * scale, '(h w) c -> 1 c h w', h=h))

        self.val_psnr(rgb_pred, rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()

        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred * 2 - 1, -1, 1),
                           torch.clip(rgb_gt * 2 - 1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test:  # save test image to disk
            idx = batch['img_idxs']
            self.save_seg(self.save_image_trans(
                results['rgb'][:, :self.model._N_COLOR_CH], f'{idx:03d}.png'),
                results['rgb'][:, self.model._N_COLOR_CH:], f'{idx:03d}_s.png')
            self.save_depth(results['depth'], f'{idx:03d}_d.png')
            if not self.current_epoch:
                self.save_image_trans(batch['rgb'][:, :self.model._N_COLOR_CH], f'{idx:03d}_gt.png')

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def save_image_trans(self, rays, name):
        base, ext = os.path.splitext(name)
        scale = torch.as_tensor((255.0, 128.0, 128.0, 128.0, 128.0), dtype=rays.dtype, device=rays.device)
        L, a, b, ta, tb = torch.unbind(rays * scale, dim=-1)
        img = self.save_image(L, a, b, name)
        self.save_image(L, ta, tb, f'{base}_t{ext}')
        return img

    def save_image(self, L, a, b, name):
        w, h = self.train_dataset.img_wh
        L = torch.clamp(L.round(), 0, 255).cpu().numpy().astype(np.uint8)
        a = torch.clamp(a.round(), -128, 127).cpu().numpy().astype(np.int8).view(np.uint8)
        b = torch.clamp(b.round(), -128, 127).cpu().numpy().astype(np.int8).view(np.uint8)
        lab = rearrange(np.stack((L, a, b), axis=1), '(h w) c -> h w c', w=w, h=h)
        lab = Image.fromarray(lab, mode='LAB')
        # Create sRGB ICC profile and convert image to sRGB
        lab_icc = ImageCms.createProfile('LAB', colorTemp=6500)
        # Create sRGB ICC profile and convert image to sRGB
        srgb_icc = ImageCms.createProfile('sRGB')
        img = ImageCms.profileToProfile(lab, lab_icc, srgb_icc, outputMode='RGB')
        img.save(os.path.join(self.val_dir, name))
        return np.asarray(img)

    def save_depth(self, depth, name):
        w, h = self.train_dataset.img_wh
        depth = depth2img(rearrange(depth.cpu().numpy(), '(h w) -> h w', h=h))
        imageio.imsave(os.path.join(self.val_dir, name), depth)
        return depth

    def save_seg(self, img, seg_logit, name):
        w, h = self.train_dataset.img_wh
        # print(name, seg_logit.min(), seg_logit.max())

        seg = rearrange(seg_logit.argmax(dim=1).cpu().numpy(), '(h w) -> h w', w=w, h=h)

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(self.palette):
            color_seg[seg == label, :] = color

        # from IPython import embed; embed(header='debug vis')
        color_seg = img * 0.5 + color_seg * 0.5
        color_seg = color_seg.astype(np.uint8)

        # save the results
        imageio.imsave(os.path.join(self.val_dir, name), color_seg)
        return color_seg


def main(hparams):
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      # check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16,
                      # gradient_clip_val=0.5
                      )

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only:  # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs - 1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs - 1}_slim.ckpt')

    if (not hparams.no_save_test) and \
            hparams.dataset_name == 'nsvf' and \
            'Synthetic' in hparams.root_dir:  # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)


if __name__ == '__main__':
    main(get_opts())
