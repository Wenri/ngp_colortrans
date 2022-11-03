import time
import warnings

import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from kornia.color import lab_to_rgb
from scipy.spatial.transform import Rotation as R

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays
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


class NGPGUI:
    _DEBUG_TRIG = False
    _N_CLS = 11

    def __init__(self, hparams, K, img_wh, radius=2.5):
        self.hparams = hparams
        rgb_act = 'None' if self.hparams.use_exposure else None
        self.model = NGP(scale=hparams.scale, rgb_act=rgb_act).cuda()
        load_ckpt(self.model, hparams.ckpt_path)

        self.cam = OrbitCamera(K, img_wh, r=radius)
        self.W, self.H = img_wh
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0

        self.register_dpg()

    def render_cam(self, cam):
        t = time.time()
        directions = get_ray_directions(cam.H, cam.W, cam.K, device='cuda')
        rays_o, rays_d = get_rays(directions, torch.cuda.FloatTensor(cam.pose))

        # TODO: set these attributes by gui
        if self.hparams.dataset_name in ['colmap', 'nerfpp']:
            exp_step_factor = 1 / 256
        else:
            exp_step_factor = 0

        trans_code = torch.zeros(11, device='cuda')
        for cls in range(NGPGUI._N_CLS):
            trans_code[cls] = dpg.get_value(f'_c{cls}')
        results = render(self.model, rays_o, rays_d,
                         **{'test_time': True,
                            # 'to_cpu': True, 'to_numpy': True,
                            'T_threshold': 1e-2,
                            # 'exposure': torch.cuda.FloatTensor([dpg.get_value('_exposure')]),
                            'max_samples': 100,
                            'trans_net_p1': trans_code,
                            'exp_step_factor': exp_step_factor})

        rgb = results["rgb"]
        scale = torch.as_tensor((100.0, 128.0, 128.0), dtype=rgb.dtype, device=rgb.device)
        rgb = torch.concat((rgb[:, :1], rgb[:, 3:5]), dim=1)
        rgb = lab_to_rgb(rearrange(rgb * scale, "(h w) c -> 1 c h w", h=self.H))
        rgb = np.ascontiguousarray(rearrange(rgb.squeeze(0), "c h w -> h w c").cpu().numpy())
        if not NGPGUI._DEBUG_TRIG:
            plt.imshow(rgb)
            plt.show()
            NGPGUI._DEBUG_TRIG = True
        depth = rearrange(results["depth"], "(h w) -> h w", h=self.H).cpu().numpy()
        torch.cuda.synchronize()
        self.dt = time.time() - t
        self.mean_samples = results['total_samples'] / len(rays_o)

        if self.img_mode == 0:
            return rgb
        elif self.img_mode == 1:
            return depth2img(depth).astype(np.float32) / 255.0

    def register_dpg(self):
        dpg.create_context()
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

        _set_default = True
        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=200, height=150):
            for cls in range(NGPGUI._N_CLS):
                dpg.add_slider_float(label=f"c{cls}",
                                     default_value=self.model.trans_net_p1[cls].item() if _set_default else 0.,
                                     min_value=-2, max_value=2, tag=f"_c{cls}")
            dpg.add_button(label="show depth", tag="_button_depth",
                           callback=callback_depth)
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

    def render(self):
        while dpg.is_dearpygui_running():
            dpg.set_value("_texture", self.render_cam(self.cam))
            dpg.set_value("_log_time", f'Render time: {1000 * self.dt:.2f} ms')
            dpg.set_value("_samples_per_ray", f'Samples/ray: {self.mean_samples:.2f}')
            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    hparams = get_opts()
    kwargs = {'root_dir': hparams.root_dir,
              'downsample': hparams.downsample,
              'read_meta': False}
    dataset = dataset_dict[hparams.dataset_name](**kwargs)

    NGPGUI(hparams, dataset.K, dataset.img_wh).render()
    dpg.destroy_context()
