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
        self._hist_loss = torch.nn.KLDivLoss()

    def forward(self, target, results, **kwargs):
        tuv, ruv = target[..., 1:3], results[..., 1:3]
        dhuv = (tuv.mean() - ruv.mean()) ** 2 * 1e-5

        sptuv, spruv = spatial_gradient(tuv, normalized=False), spatial_gradient(ruv, normalized=False)
        spmask = torch.all(torch.lt(spruv.abs(), 0.5), dim=2, keepdim=True).expand_as(spruv)
        dhuv += torch.mean((sptuv[spmask] - spruv[spmask]) ** 2)

        # tuv, ruv = torch.nn.functional.avg_pool2d(tuv, (2, 2)), torch.nn.functional.avg_pool2d(ruv, (2, 2))
        # tuv, ruv = rearrange(tuv, 'b c h w -> (b h w) c'), rearrange(ruv, 'b c h w -> (b h w) c')
        # thuv, rhuv = self._hist_func(tuv), self._hist_func(ruv)
        # dhuv += self._hist_loss(rhuv, thuv) * 1e-1

        return dhuv


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion

    def _yuv_loss(self, target_yuv, results_yuv):
        ty, tuv = target_yuv[..., 0], target_yuv[..., 1:3]
        ry, ruv = results_yuv[..., 0], results_yuv[..., 1:3]
        dy = (ty - ry) ** 2 * 1e-1
        duv = (tuv - ruv) ** 2 * 1e-3

        return torch.cat((dy.unsqueeze(-1), duv), dim=-1)

    def forward(self, results, target, **kwargs):
        o = results['opacity'] + 1e-10

        d = {
            'rgb': self._yuv_loss(target_yuv=target['rgb'], results_yuv=results['rgb']),
            # encourage opacity to be either 0 or 1 to avoid floater
            'opacity': self.lambda_opacity * (-o * torch.log(o)),
        }

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                              DistortionLoss.apply(results['ws'], results['deltas'],
                                                   results['ts'], results['rays_a'])

        return d
