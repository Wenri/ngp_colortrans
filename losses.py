import torch
import torch.nn.functional as F
import vren
from einops import rearrange
from kornia.filters import spatial_gradient
from torch import nn

from misc.differentiable_histogram import MultivariateGaussianHistogram


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """

    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
         ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class HistLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self._hist_func = GaussianHistogram(bins=100, min=-0.436, max=0.436, sigma=1e-2)
        self._hist_func = MultivariateGaussianHistogram(bins=32, min=-0.615, max=0.615, sigma=(1e-2, 1e-2))
        self._hist_loss = torch.nn.KLDivLoss(reduction='none')
        self._l1_loss = torch.nn.L1Loss(reduction='none')

    def forward(self, results, target, **kwargs):
        if self:
            raise NotImplementedError

        tuv, ruv = rearrange(target[..., 1:3], 'b h w c -> b c h w'), rearrange(results[..., 1:3], 'b h w c -> b c h w')
        dhuv = self._l1_loss(input=ruv.mean(), target=tuv.mean()) * 1e-3

        sptuv, spruv = spatial_gradient(tuv, normalized=True), spatial_gradient(ruv, normalized=True)
        sptuv, spruv = rearrange(sptuv, 'b c o h w -> b h w c o'), rearrange(spruv, 'b c o h w -> b h w c o')
        dhuv = dhuv + self._l1_loss(input=spruv, target=sptuv)
        # spmask = torch.all(torch.lt(spruv.abs(), 1.8), dim=-1)
        # spmask = torch.all(spmask, dim=-1)
        # dhuv = dhuv + (sptuv[spmask] - spruv[spmask]) ** 2

        # tuv, ruv = torch.nn.functional.avg_pool2d(tuv, (2, 2)), torch.nn.functional.avg_pool2d(ruv, (2, 2))
        # tuv, ruv = rearrange(tuv, 'b c h w -> (b h w) c'), rearrange(ruv, 'b c h w -> (b h w) c')
        # thuv, rhuv = self._hist_func(tuv), self._hist_func(ruv)
        # dhuv = self._hist_loss(input=rhuv / rhuv.sum(), target=thuv / thuv.sum()) * 1e-2
        # dhuv = self._l1_loss(input=rhuv, target=thuv) * 1e-2
        return dhuv


class NeRFLoss(nn.Module):
    _EPSILON = torch.finfo(torch.float32).eps

    def __init__(self, n_color_ch, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self._l1_loss = torch.nn.HuberLoss(reduction='none', delta=0.1)
        self._l2_loss = torch.nn.MSELoss(reduction='none')
        self._n_color_ch = n_color_ch
        # self.trans_w = torch.nn.Linear(826, 2)

    def setup_sem_ind(self, sem_ind, from_points, to_points, flow, coeffs):
        self.register_buffer('sem_ind', sem_ind)
        self.register_buffer('from_points', from_points)
        self.register_buffer('to_points', to_points)
        self.register_buffer('flow', flow)
        self.register_buffer('coeffs', coeffs)

    def _U(self, x: torch.Tensor):
        return x * torch.where(torch.lt(x, self._EPSILON), 0., torch.log(x) / 2)

    def _calculate_f(self, coeffs, x, y):
        w = coeffs[:-3]
        a1, ax, ay = coeffs[-3:]
        # The following may use too much RAM:
        points = self.to_points.to(dtype=torch.float64)
        distances = self._U(torch.square(points[:, 0] - x[..., None]) + torch.square(points[:, 1] - y[..., None]))
        distances = (w * distances).sum(axis=-1)
        return a1 + ax * x + ay * y + distances

    def _trans_ab(self, img):
        a, b = torch.unbind(img, dim=1)
        a, b = self._calculate_f(self.coeffs[:, 0], a, b), self._calculate_f(self.coeffs[:, 1], a, b)
        return torch.stack((a, b), dim=1)

    def _seg_loss(self, results_seg, target_seg):
        seg = results_seg[..., self._n_color_ch:]
        n_sem = seg.shape[1] - 1
        target_seg = torch.softmax(target_seg, dim=1, dtype=torch.float32)
        target_seg_n = target_seg[..., self.sem_ind[:n_sem]]
        target_seg_nr = target_seg_n.max(dim=1)
        target_seg_o = target_seg[..., self.sem_ind[n_sem:]]
        target_seg_or = torch.sum(target_seg_o, dim=1)

        target_is_n = torch.ge(target_seg_nr.values, target_seg_or)
        target_idx = torch.where(target_is_n, target_seg_nr.indices, n_sem)
        target_conf = torch.where(target_is_n, target_seg_nr.values, target_seg_or)
        selected = torch.ge(target_conf, 0.5)

        loss = target_conf[selected] * F.cross_entropy(seg[selected], target_idx[selected], reduction='none')
        return loss * 1e-1

    def _lab_loss(self, results_ab, target_ab):
        weight = 1e-1
        n_ch = 2
        results_ab = results_ab[..., 3:self._n_color_ch]
        target_ab = target_ab[..., 3:]
        loss = []
        for idx in range(0, results_ab.shape[1], n_ch):
            # loss.append(self._l1_loss(input=results_ab[..., idx:idx + n_ch],
            #                           target=target_ab[..., idx:idx + n_ch]) * weight)
            # ref_ab = self._trans_ab(target_ab[..., idx:idx + n_ch]).to(dtype=results_ab.dtype)
            rt = self._trans_ab(results_ab[..., idx:idx + n_ch]).to(dtype=target_ab.dtype)
            # distance = rearrange(results_ab[..., idx:idx + n_ch], 'b c -> b 1 c') - self.to_points
            # distance = torch.sum(torch.square(distance), dim=-1)
            # flowd = torch.matmul(distance, self.flow)
            # flowd = flowd / torch.sum(self.flow, dim=0)
            loss.append(self._l1_loss(rt, target=target_ab[..., idx:idx + n_ch]) * weight)

        return torch.cat(loss, dim=1)

    def _rgb_loss(self, results_rgb, target_rgb):
        return self._l2_loss(input=results_rgb[..., :3], target=target_rgb[..., :3])

    def forward(self, results, target, **kwargs):
        o = results['opacity'] + self._EPSILON

        d = {
            'rgb': self._rgb_loss(results_rgb=results['rgb'], target_rgb=target['rgb']),
            'trans': self._lab_loss(results_ab=results['rgb'], target_ab=target['rgb']),
            'seg': self._seg_loss(results_seg=results['rgb'], target_seg=target['seg']),
            # encourage opacity to be either 0 or 1 to avoid floater
            'opacity': self.lambda_opacity * (-o * torch.log(o)),
        }

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                              DistortionLoss.apply(results['ws'], results['deltas'],
                                                   results['ts'], results['rays_a'])

        return d
