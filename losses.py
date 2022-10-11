import torch
from kornia.color import rgb_to_yuv
from torch import nn
import vren
from einops import rearrange

from misc.differentiable_histogram import GaussianHistogram
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


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self._hist_u = GaussianHistogram(bins=100, min=-0.436, max=0.436, sigma=1e-2)
        self._hist_v = GaussianHistogram(bins=100, min=-0.615, max=0.615, sigma=1e-2)

    def _yuv_loss(self, target_yuv, results_yuv):
        ty, tu, tv = target_yuv[:, 0], target_yuv[:, 1], target_yuv[:, 2]
        ry, ru, rv = results_yuv[:, 0], results_yuv[:, 1], results_yuv[:, 2]
        dy = (ty - ry) ** 2
        du = (tu - ru) ** 2 * 1e-8 # + tu * 1e-2
        dv = (tv - rv) ** 2 * 1e-8 # + tv * 1e-2

        thu, thv = self._hist_u(tu), self._hist_v(tv)
        rhu, rhv = self._hist_u(ru), self._hist_v(rv)

        return torch.stack((dy, du, dv), dim=-1)

    def forward(self, results, target, **kwargs):
        d = {}

        d['rgb'] = self._yuv_loss(
            target_yuv=rearrange(rgb_to_yuv(rearrange(target['rgb'], 'b c -> b c 1 1')), 'b c 1 1 -> b c'),
            results_yuv=rearrange(rgb_to_yuv(rearrange(results['rgb'], 'b c -> b c 1 1')), 'b c 1 1 -> b c'))

        o = results['opacity'] + 1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = self.lambda_opacity * (-o * torch.log(o))

        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                              DistortionLoss.apply(results['ws'], results['deltas'],
                                                   results['ts'], results['rays_a'])

        return d
