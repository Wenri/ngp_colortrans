import torch
from kornia.color import rgb_to_yuv, rgb_to_yuv420
from kornia.filters import spatial_gradient
from torch import nn
import vren
from einops import rearrange

from misc.differentiable_histogram import GaussianHistogram, MultivariateGaussianHistogram
from misc.rgb_lab_formulation_pytorch import rgb_to_lab


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
        tuv, ruv = rearrange(target[..., 1:3], 'b h w c -> b c h w'), rearrange(results[..., 1:3], 'b h w c -> b c h w')
        dhuv = self._l1_loss(input=ruv.mean(dim=(2, 3)), target=tuv.mean(dim=(2, 3))) * 1e-3

        sptuv, spruv = spatial_gradient(tuv, normalized=True), spatial_gradient(ruv, normalized=True)
        sptuv, spruv = rearrange(sptuv, 'b c o h w -> b h w c o'), rearrange(spruv, 'b c o h w -> b h w c o')
        dhuv = rearrange(dhuv, 'b c -> b 1 1 c 1')
        dhuv = dhuv + self._l1_loss(input=spruv, target=sptuv) * 1e-1
        # spmask = torch.any(torch.le(spruv.abs(), 1.0), dim=-1)
        # spmask = torch.any(spmask, dim=-1)
        # dhuv = dhuv + self._l1_loss(input=spruv[spmask], target=sptuv[spmask])

        # tuv, ruv = torch.nn.functional.avg_pool2d(tuv, (2, 2)), torch.nn.functional.avg_pool2d(ruv, (2, 2))
        # tuv, ruv = rearrange(tuv, 'b c h w -> (b h w) c'), rearrange(ruv, 'b c h w -> (b h w) c')
        # thuv, rhuv = self._hist_func(tuv), self._hist_func(ruv)
        # dhuv = dhuv + self._l1_loss(input=rhuv, target=thuv).mean() * 1e-2
        return dhuv


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion

    def _yuv_loss(self, results_yuv, target_yuv):
        ry, ty = results_yuv[..., 0], target_yuv[..., 0]
        ruv, tuv = results_yuv[..., 1:3], target_yuv[..., 1:3]
        dy = (ty - ry) ** 2
        duv = (tuv - ruv) ** 2 * 1e-5
        return torch.cat((dy.unsqueeze(-1), duv), dim=-1)

    def forward(self, results, target, **kwargs):
        o = results['opacity'] + 1e-10

        d = {
            'rgb': self._yuv_loss(results_yuv=results['rgb'], target_yuv=target['rgb']),
            # encourage opacity to be either 0 or 1 to avoid floater
            'opacity': self.lambda_opacity * (-o * torch.log(o)),
        }

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                              DistortionLoss.apply(results['ws'], results['deltas'],
                                                   results['ts'], results['rays_a'])

        return d
