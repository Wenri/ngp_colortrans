import asyncio
import os
import subprocess
import tempfile
import time
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np
import torch
from PIL import Image
from apex.optimizers import FusedAdam
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.spatial.transform import Rotation as R
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays
from losses import NeRFLoss
from models.custom_functions import total_variation_loss
from models.networks import NGP
from models.rendering import render
from opt import get_opts
from train import depth2img
from utils import load_ckpt

warnings.filterwarnings("ignore")


class OrbitCamera:
    def __init__(self, K, img_wh, r):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r
        self.center = np.zeros(3)
        self.rot = np.eye(3)

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(0.05 * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NGPGUI(LightningModule):
    def __init__(self, hparams, K, img_wh, radius=2.5):
        super().__init__()
        self.manual_callback_management = False
        self.save_hyperparameters(hparams)
        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=hparams.scale, rgb_act=rgb_act, bil_grids=True).cuda()
        load_ckpt(self.model, hparams.ckpt_path)

        self.cam = OrbitCamera(K, img_wh, r=radius)
        self.W, self.H = img_wh
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0
        self.bilgrid3d_tv_loss_mult = 1e-2
        self.editing = asyncio.Event()
        self.editing_target = None
        self.texture = None

        self.register_dpg()

    def setup(self, stage):
        directions, grid = get_ray_directions(self.cam.H, self.cam.W, self.cam.K, return_uv=True)
        rays_o, rays_d = get_rays(directions, torch.FloatTensor(self.cam.pose))
        editing_target = rearrange(self.editing_target, "h w c -> (h w) c")
        self.train_dataset = TensorDataset(rays_o, rays_d, grid, editing_target)

    def configure_optimizers(self):
        # define additional parameters
        net_params = [p for n, p in self.model.bil_grids.named_parameters()]
        net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        net_sch = CosineAnnealingLR(net_opt, self.hparams.num_epochs, self.hparams.lr / 30)
        return [net_opt], [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=0,
                          batch_size=16384, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=0,
                          batch_size=16384)

    def forward(self, rays_o, rays_d, test_time=True) -> Any:
        # TODO: set these attributes by gui
        if self.hparams.dataset_name in ['colmap', 'nerfpp']:
            exp_step_factor = 1 / 2 ** 10
        else:
            exp_step_factor = 0

        return render(self.model, rays_o, rays_d, **{
            'test_time': test_time, 'to_cpu': test_time, 'to_numpy': test_time,
            'T_threshold': 1e-2,
            'alpha': dpg.get_value('_alpha'),
            'max_samples': 100,
            'exp_step_factor': exp_step_factor})

    def render_cam(self, cam):
        t = time.time()

        directions = get_ray_directions(cam.H, cam.W, cam.K, device='cuda')
        rays_o, rays_d = get_rays(directions, torch.cuda.FloatTensor(cam.pose))
        results = self(rays_o, rays_d)

        rgb = rearrange(results["rgb"], "(h w) c -> h w c", h=self.H)
        depth = rearrange(results["depth"], "(h w) -> h w", h=self.H)
        torch.cuda.synchronize()
        self.dt = time.time() - t
        self.mean_samples = results['total_samples'] / depth.size

        if self.img_mode == 0:
            return rgb
        elif self.img_mode == 1:
            return depth2img(depth).astype(np.float32) / 255.0

    def register_dpg(self):
        dpg.create_context()
        if self.manual_callback_management:
            dpg.configure_app(manual_callback_management=True)
        dpg.create_viewport(title="ngp_pl", width=self.W, height=self.H, resizable=False)

        ## register texture ##
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture")

        ## register window ##
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        def callback_depth(sender, app_data):
            self.img_mode = 1 - self.img_mode

        def callback_edit(sender, app_data):
            self.editing.set()

        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=200, height=150):
            dpg.add_slider_float(label="alpha", default_value=1,
                                 min_value=0, max_value=1, tag="_alpha")
            dpg.add_button(label="show depth", tag="_button_depth", callback=callback_depth)
            dpg.add_button(label="edit view", tag="_button_edit", callback=callback_edit)
            dpg.add_separator()
            dpg.add_text('no data', tag="_log_time")
            dpg.add_text('no data', tag="_samples_per_ray")

        ## register camera handler ##
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.orbit(app_data[1], app_data[2])

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.scale(app_data)

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.pan(app_data[1], app_data[2])

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        ## Avoid scroll bar in the window ##
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        ## Launch the gui ##
        dpg.setup_dearpygui()
        dpg.set_viewport_small_icon("assets/icon.png")
        dpg.set_viewport_large_icon("assets/icon.png")
        dpg.show_viewport()

    async def render(self):
        while dpg.is_dearpygui_running():
            if self.manual_callback_management:
                dpg.run_callbacks(dpg.get_callback_queue())
            if self.editing.is_set():
                await self.edit()
                self.editing.clear()
            self.update(self.render_cam(self.cam))

    def update(self, texture):
        dpg.set_value("_texture", texture)
        dpg.set_value("_log_time", f'Render time: {1000 * self.dt:.2f} ms')
        dpg.set_value("_samples_per_ray", f'Samples/ray: {self.mean_samples:.2f}')
        dpg.render_dearpygui_frame()

    async def edit(self):
        with ExitStack() as stack:
            dpg.set_value('_alpha', 1.0)
            f = tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=Path(
                "logs", hparams.dataset_name, hparams.exp_name))
            stack.callback(os.unlink, f.name)
            stack.enter_context(f)
            texture = self.render_cam(self.cam)
            img = texture * 255
            img = Image.fromarray(img.astype(np.uint8))
            img.save(f, format='PNG')
            f.close()
            self.texture = torch.from_numpy(texture)
            subprocess.run(['/snap/bin/gimp', f.name])
            img = Image.open(f.name)
        return await self.bil_opt(np.asanyarray(img))

    async def bil_opt(self, img):
        self.editing_target = torch.from_numpy(img) / 255.

        callbacks = [TQDMProgressBar(refresh_rate=1)]

        logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                                   name=hparams.exp_name,
                                   default_hp_metric=False)

        trainer = Trainer(max_epochs=2,
                          check_val_every_n_epoch=1,
                          limit_val_batches=1,
                          callbacks=callbacks,
                          logger=logger,
                          enable_model_summary=False,
                          accelerator='gpu',
                          devices=hparams.num_gpus,
                          strategy="ddp",
                          num_sanity_val_steps=-1 if hparams.val_only else 0,
                          precision=16)

        trainer.fit(self)
        self.model.cuda()

    def validation_step(self, *args: Any, **kwargs: Any):
        texture = self.render_cam(self.cam)
        self.update(texture)
        self.texture = torch.from_numpy(texture)

    def training_step(self, batch, batch_nb, *args):
        rays_o, rays_d, grid, target = batch
        target = {
            'rgb': target,
        }
        results = self(rays_o, rays_d, test_time=False)

        loss_d = self.loss(results, target)

        total_loss = 0.

        bilagrid_3d = self.model.bil_grids
        # Add TV loss to CP factors.
        for i in range(1, bilagrid_3d.num_facs):
            fac = bilagrid_3d.get_parameter(f'fac_{i}')
            total_loss += self.bilgrid3d_tv_loss_mult * total_variation_loss(fac)

        loss_d['tv_bilgrids3d'] = total_loss
        loss = sum(lo.mean() for lo in loss_d.values())

        grid = grid.to(device=self.texture.device, dtype=torch.long)
        rgb = results['rgb'].detach().to(device=self.texture.device)
        self.update(self.texture.index_put_((grid[:, 1], grid[:, 0]), rgb).numpy())

        return loss


if __name__ == "__main__":
    hparams = get_opts()
    kwargs = {'root_dir': hparams.root_dir,
              'downsample': hparams.downsample,
              'read_meta': False}
    dataset = dataset_dict[hparams.dataset_name](**kwargs)

    asyncio.run(NGPGUI(hparams, dataset.K, dataset.img_wh).render())
    dpg.destroy_context()
