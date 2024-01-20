import gin
import tensorly as tl
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.utils import set_kwargs

tl.set_backend('pytorch')
from tensorly.decomposition import parafac
from torchcubicspline import NaturalCubicSpline


@gin.configurable
class CubicSplines(torch.nn.Module):
    """Cubic splines. Each input channel corresponds to an independent curve."""

    # The number of cubic curve parameters.
    N_curve_params = 4
    N_knots = 20

    def __init__(self, in_dim, **kwargs):
        '''
        :param in_dim: number of input channels.
        '''
        super(CubicSplines, self).__init__()
        set_kwargs(self, kwargs)

        self.num_channels = in_dim

        coeffs = torch.zeros(self.num_channels, self.N_curve_params, self.N_knots - 1, 1)
        knots_t = torch.zeros(self.num_channels, self.N_knots)
        # Curve coefficients.
        self.register_buffer('coeffs', coeffs)
        # t values of knots.
        self.register_buffer('knots_t', knots_t)

        self._curves_from_coeffs()
        self.register_load_state_dict_post_hook(self._curves_from_coeffs)

    def init_curves(self, init_coeffs):
        '''
        :param init_coeffs: a list of curve coefficients (computed from color_cdf_spline_coeffs).
        '''
        coeffs = torch.stack([torch.stack(init_coeffs[i][1:], dim=0) for i in range(self.num_channels)], dim=0)
        knots_t = torch.stack([init_coeffs[i][0] for i in range(self.num_channels)], dim=0)
        self.coeffs = coeffs
        self.knots_t = knots_t
        self._curves_from_coeffs()

    def _curves_from_coeffs(self, *unused_args):
        self.splines = []
        for i in range(self.num_channels):
            # Cubic spline encapsulation
            # See: https://github.com/patrick-kidger/torchcubicspline/blob/master/torchcubicspline/interpolate.py#L181C45-L181C90
            spline = NaturalCubicSpline([self.knots_t[i]] + list(self.coeffs[i]))
            self.splines.append(spline)

    def forward(self, x):
        '''
        :param x: (..., in_dim)
        '''
        return torch.stack([self.splines[j].evaluate(x[..., j]).squeeze() for j in range(self.num_channels)], dim=-1)


@gin.configurable
class Rgb2Gray(nn.Module):
    LUMA = 'luma'
    MLP = 'mlp'
    COLOR_CDF = 'color_cdf'
    CONSTANT_ZERO = 'constant_zero'

    mlp_depth = 2
    mlp_width = 8

    class ScaledTanh(nn.Module):
        def __init__(self, s=2.0):
            super().__init__()
            self.scaler = s

        def forward(self, x):
            return torch.tanh(self.scaler * x)

    def __init__(self, rgb2gray_fn, **kwargs):
        super(Rgb2Gray, self).__init__()
        set_kwargs(self, kwargs)

        if rgb2gray_fn == Rgb2Gray.COLOR_CDF:
            self.rgb_warp = CubicSplines(3)
            color_cdf_spline_coeffs = kwargs.get('color_cdf_spline_coeffs', None)
            if color_cdf_spline_coeffs is not None:
                self.rgb_warp.init_curves(color_cdf_spline_coeffs)
            self.register_buffer('rgb2gray_weight', torch.Tensor([[1. / 3., 1. / 3., 1. / 3.]]))
            self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2. - 1.

        elif rgb2gray_fn == Rgb2Gray.MLP:
            rgb2gray_mlp_linear = lambda l: nn.Linear(self.mlp_width, self.mlp_width if l < self.mlp_depth - 1 else 1)
            rgb2gray_mlp_actfn = lambda _: nn.ReLU(inplace=True)
            self.rgb2gray = nn.Sequential(
                *([nn.Linear(3, self.mlp_width)] + \
                  [nn_module(l) for l in range(1, self.mlp_depth) for nn_module in
                   [rgb2gray_mlp_actfn, rgb2gray_mlp_linear]] + \
                  [Rgb2Gray.ScaledTanh(2.)]))

        elif rgb2gray_fn == Rgb2Gray.LUMA:
            # Weights of BT601/BT470 RGB-to-gray.
            self.register_buffer('rgb2gray_weight', torch.Tensor([[0.299, 0.587, 0.114]]))
            self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2. - 1.

        elif rgb2gray_fn == Rgb2Gray.CONSTANT_ZERO:
            self.register_buffer('rgb2gray_weight', torch.Tensor([[0., 0., 0.]]))
            self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T)

        else:
            raise ValueError(f'Cannot recognize {rgb2gray_fn}')

    def forward(self, rgb):
        '''
        :return gray-scale values in [-1, 1]
        '''
        return self.rgb2gray(rgb)


def color_affine_transform(affine_mats, rgb):
    '''
    Apply per-pixel color affine transformations.
    :param affine_mats: affine transformation matrices (..., 3, 4).
    :param rgb: pixel RGB values (..., 3).
    :return: color transformed image (..., 3).
    '''
    return torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1) + affine_mats[..., 3]


def slice(bil_grids, xy, rgb, grid_idx):
    '''
    Slice batch of bilateral grids. Suppose N is number of bilateral grids. The slicing supports two cases:
        1) Slice all bilateral grids. Supporting shapes of xy: (P, height, width, 2).
           In this case, grid_idx should be [0, 0, ....., 0, 1, 1, ....., 1, N, N, ....., N].
                                             <-- P // N -->  <-- P // N -->  <-- P // N -->
        2) Slice bilateral grids indexed by grid_idx. Supporting shapes of xy: (chunk_size, 2) or (height, width, 2).

    :param bil_grids: the bilateral grids.
    :param xy: the x-y coordinates (..., 2).
    :param rgb: the corresponding RGB values (..., 3).
    :param grid_idx: grid indices for slicing (..., 1).
    '''
    sh_ = rgb.shape

    grid_idx_unique = torch.unique(grid_idx)
    if len(grid_idx_unique) == 1:
        # All pixels are from a single view.
        # possible shapes of xy: # (chunk_size, 2) or (height, width, 2)
        grid_idx = grid_idx_unique  # (1,)
        xy = xy.unsqueeze(0)  # (1, ..., 2)
        rgb = rgb.unsqueeze(0)  # (1, ..., 3)
    else:
        grid_idx_unique, unique_counts = torch.unique(grid_idx, return_counts=True, dim=0)
        if len(grid_idx_unique) == bil_grids.grids.shape[0] and len(torch.unique(unique_counts)) == 1:
            # Pixels are sampled from every view. Requires Config.uniform_batching_from_all_views = True.
            unique_counts = unique_counts.tolist()
            grid_idx = None
            # Gather pixels by their camera index.
            # possible shape of xy: (num_patches, patch_size, patch_size, 2)
            # reshaped into (num_cams, num_patches // num_cams, patch_size, patch_size, 2)
            xy = torch.stack(torch.split(xy, unique_counts, dim=0), dim=0)
            rgb = torch.stack(torch.split(rgb, unique_counts, dim=0), dim=0)
        else:
            # Pixels are randomly sampled from different views.
            # possible shapes of xy: # (chunk_size, 2) or (num_patches, patch_size, patch_size, 2)
            if len(grid_idx.shape) == 4:
                grid_idx = grid_idx[:, 0, 0, 0]  # (num_cams,)
            elif len(grid_idx.shape) == 2:
                grid_idx = grid_idx[:, 0]  # (num_cams,)
            else:
                raise ValueError(f'The input to bilateral grid slicing is not supported yet.')

    affine_mats = bil_grids(xy, rgb, grid_idx)
    rgb = color_affine_transform(affine_mats, rgb)

    return {
        'rgb': rgb.reshape(*sh_),
        'rgb_affine_mats': affine_mats.reshape(*sh_[:-1], affine_mats.shape[-2], affine_mats.shape[-1])
    }


@gin.configurable
class BilateralGrid(nn.Module):
    grid_width = 16  # number of grids in x-axis.
    grid_height = 16  # number of grids in y-axis.
    grid_depth = 8  # number of grids in z-axis.
    learn_gray = False  # If True, the gray value will be learned.
    spatial_only = False  # If True, the gray value will be always mapped to 0.

    def __init__(self, num, **kwargs):
        """
        :param num: number of bilateral grids (= # of camera views).
        """
        super(BilateralGrid, self).__init__()
        set_kwargs(self, kwargs)

        # Initialize grids.
        grid = self.init_identity_grid()
        self.grids = nn.Parameter(grid.tile(num, 1, 1, 1, 1))  # (N, 12, D, H, W)

        if self.spatial_only:
            self.register_buffer('rgb2gray_weight', torch.Tensor([[0, 0, 0]]))
            self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T)
        elif self.learn_gray:
            self.rgb2gray = nn.Sequential(
                nn.Linear(3, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, 1),
                Rgb2Gray.ScaledTanh(2.))
        else:
            # Weights of BT601 RGB-to-gray.
            self.register_buffer('rgb2gray_weight', torch.Tensor([[0.299, 0.587, 0.114]]))
            self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2. - 1.

    def init_identity_grid(self):
        grid = torch.tensor([1., 0, 0, 0, 0, 1., 0, 0, 0, 0, 1., 0, ]).float()
        grid = grid.repeat([self.grid_depth * self.grid_height * self.grid_width, 1])  # (D * H * W, 12)
        grid = grid.reshape(1, self.grid_depth, self.grid_height, self.grid_width, -1)  # (1, D, H, W, 12)
        grid = grid.permute(0, 4, 1, 2, 3)  # (1, 12, D, H, W)
        return grid

    def forward(self, grid_xy, rgb, idx=None):
        """
        Bilateral grid slicing.
        :param grid_xy: x-y coordinates (num_cams, ..., 2).
                        When not using 5D input, the `idx` parameter should be specified.
        :param rgb: (num_cams, ..., 3).
        :param idx: camera indices (num_cams,).
        :return: affine matrices (num_cams, ..., 3, 4).
        """
        grids = self.grids
        input_ndims = len(grid_xy.shape)
        assert len(rgb.shape) == input_ndims

        if input_ndims > 1 and input_ndims < 5:
            # Convert input into 5D
            for i in range(5 - input_ndims):
                grid_xy = grid_xy.unsqueeze(1)
                rgb = rgb.unsqueeze(1)
            assert idx is not None
        elif input_ndims != 5:
            raise ValueError('Bilateral grid slicing only takes either 2D, 3D, 4D and 5D inputs')

        grids = self.grids
        if idx is not None:
            grids = grids[idx]
        assert grids.shape[0] == grid_xy.shape[0]

        # Generate slicing coordinates.
        grid_xy = (grid_xy - 0.5) * 2  # Rescale to [-1, 1].
        grid_z = self.rgb2gray(rgb)
        grid_xyz = torch.cat([grid_xy, grid_z], dim=-1)  # (N, num_patches, h, w, 3)

        affine_mats = F.grid_sample(grids, grid_xyz, mode='bilinear', align_corners=True,
                                    padding_mode='border')  # (N, 12, num_patches, h, w)
        affine_mats = affine_mats.permute(0, 2, 3, 4, 1)  # (N, num_patches, h, w, 12)
        affine_mats = affine_mats.reshape(*affine_mats.shape[:-1], 3, 4)  # (N, num_patches, h, w, 3, 4)

        for _ in range(5 - input_ndims):
            affine_mats = affine_mats.squeeze(1)

        return affine_mats


def slice3d(bil_grids3d, xyz, rgb):
    '''
    :param bil_grids3d: the 3D bilateral grids.
    :param xyz: the xyz coordinates (..., 3).
    :param rgb: the corresponding RGB values (..., 3).
    '''
    affine_mats = bil_grids3d(xyz, rgb)
    rgb = color_affine_transform(affine_mats, rgb)

    return {
        'rgb': rgb,
        'rgb_affine_mats': affine_mats
    }


@gin.configurable
class BilateralGridCP3D(nn.Module):
    grid_X = 16  # number of grids in x-axis.
    grid_Y = 16  # number of grids in y-axis.
    grid_Z = 16  # number of grids in z-axis.
    grid_W = 8  # number of grids in w-axis (gray value).
    rank = 5  # number of components
    learn_gray = False  # If True, equivalent to rgb2gray_fn = 'mlp'.
    training_rgb2gray_mlp = True  # If True, rgb2gray MLP model is trainable.
    rgb2gray_fn = 'luma'  # Function to warp RGB ('color_cdf', 'mlp', 'luma', 'constant_zero').
    init_noise_scale = 1e-6  # The noise scale of the initialized factors.

    def __init__(self, **kwargs):
        super(BilateralGridCP3D, self).__init__()
        set_kwargs(self, kwargs)

        # Initialize identity grids.
        init_grids = self.init_identity_grid()
        # Random noises are added to avoid singularity.
        init_grids = torch.randn_like(init_grids) * self.init_noise_scale + init_grids
        # Initialize grid CP factors
        _, facs = parafac(init_grids.clone().detach(), rank=self.rank)

        self.num_facs = len(facs)

        self.fac_0 = nn.Linear(facs[0].shape[0], facs[0].shape[1], bias=False)
        self.fac_0.weight = nn.Parameter(facs[0])  # (12, rank)

        for i in range(1, self.num_facs):
            fac = facs[i].T  # (rank, grid_size)
            fac = fac.view(1, fac.shape[0], fac.shape[1], 1)  # (1, rank, grid_size, 1)
            self.register_buffer(f'fac_{i}_init', fac)

            fac_resid = torch.zeros_like(fac)
            self.register_parameter(f'fac_{i}', nn.Parameter(fac_resid))

        if self.learn_gray:
            self.rgb2gray_fn = Rgb2Gray.MLP
        self.rgb2gray = Rgb2Gray(self.rgb2gray_fn,
                                 color_cdf_spline_coeffs=kwargs.get('color_cdf_spline_coeffs', None))
        if not self.training_rgb2gray_mlp:
            for param in self.rgb2gray.parameters():
                param.requires_grad = False

    def init_identity_grid(self):
        grid = torch.tensor([1., 0, 0, 0, 0, 1., 0, 0, 0, 0, 1., 0, ]).float()
        grid = grid.repeat([self.grid_W * self.grid_Z * self.grid_Y * self.grid_X, 1])
        grid = grid.reshape(self.grid_W, self.grid_Z, self.grid_Y, self.grid_X, -1)
        grid = grid.permute(4, 0, 1, 2, 3)  # (12, grid_W, grid_Z, grid_Y, grid_X)
        return grid

    def forward(self, xyz, rgb):
        """
        :param xyz: (..., 3)
        :param rgb: (..., 3)
        :return: (..., 3, 4)
        """
        sh_ = xyz.shape
        xyz = xyz.reshape(-1, 3)  # flatten (N, 3)
        rgb = rgb.reshape(-1, 3)  # flatten (N, 3)

        bound = 2
        xyz = xyz / bound

        gray = self.rgb2gray(rgb)
        xyzw = torch.cat([xyz, gray], dim=-1)  # (N, 4)
        xyzw = xyzw.transpose(0, 1)  # (4, N)
        coords = torch.stack([torch.zeros_like(xyzw), xyzw], dim=-1)  # (4, N, 2)
        coords = coords.unsqueeze(1)  # (4, 1, N, 2)

        coef = 1.
        for i in range(1, self.num_facs):
            fac = self.get_parameter(f'fac_{i}') + self.get_buffer(f'fac_{i}_init')
            coef = coef * F.grid_sample(fac, coords[[i - 1]],
                                        align_corners=True,
                                        padding_mode='border')  # [1, rank, 1, N]
        coef = coef.squeeze([0, 2]).transpose(0, 1)  # (N, rank)
        mat = self.fac_0(coef)
        return mat.reshape(*sh_[:-1], 3, 4)


def grid_sample_4d(grids, xyzw):
    """
    4D support for `torch.nn.functional.grid_sample`.
    Assumes align_corners=True, mode='bilinear', padding_mode='border'

    :param grids: (C, W, Z, Y, X)
    :param xyzw: (..., 4)
    :return: (..., C)
    """
    C, W, _, _, _ = grids.shape
    sh_ = xyzw.shape
    xyzw = xyzw.reshape(-1, sh_[-1])

    xyz = xyzw[:, :-1].reshape(1, 1, 1, -1, 3)  # (1, 1, 1, N, 3)
    w = xyzw[:, -1]  # (N,)

    # Padding: border.
    w = torch.clamp(w + 1., min=0, max=2.)

    intvl_w = 2. / (W - 1)
    w = w / intvl_w  # Rescale to [0, W - 1]
    idx_lower = torch.clamp(torch.floor(w), min=0., max=W - 2.)  # (N,)
    # Find interpolation weights (align corners)
    w_weights = w - idx_lower

    out = torch.zeros(w.shape[0], C).to(grids.device, dtype=grids.dtype)

    grids_ = grids.transpose(0, 1)  # (W, C, Z, Y, X)
    # Traverse each grid along w-axis.
    # Note: directly indexing grids by idx_lower is memory inefficient.
    for i in range(W - 1):
        # Find coordinates in the current grid.
        in_grid = idx_lower.long() == i
        if in_grid.sum() > 0:
            in_grid_idx = torch.nonzero(in_grid).squeeze(-1)  # (N',)
            xyz_ = torch.index_select(xyz, 3, in_grid_idx)  # (1, 1, 1, N', 3)
            w_weights_ = w_weights[in_grid_idx].unsqueeze(-1)  # (N', 1)
            out_lower = F.grid_sample(grids_[[i], ...], xyz_,
                                      align_corners=True,
                                      padding_mode='border')  # (1, C, 1, 1, N')
            out_lower = out_lower.squeeze(0, 2, 3).transpose(1, 0)  # (N', C)
            out_upper = F.grid_sample(grids_[[i + 1], ...], xyz_,
                                      align_corners=True,
                                      padding_mode='border')  # (1, C, 1, 1, N')
            out_upper = out_upper.squeeze(0, 2, 3).transpose(1, 0)  # (N', C)

            out_ = (1. - w_weights_) * out_lower + w_weights_ * out_upper  # (N', C)
            out[in_grid, :] = out_

    return out.reshape(*sh_[:-1], C)


@gin.configurable
class BilateralGrid3D(nn.Module):
    grid_X = 16  # number of grids in x-axis.
    grid_Y = 16  # number of grids in y-axis.
    grid_Z = 16  # number of grids in z-axis.
    grid_W = 8  # number of grids in w-axis (gray value).
    learn_gray = False  # If True, equivalent to rgb2gray_fn = 'mlp'.
    training_rgb2gray_mlp = True  # If True, rgb2gray MLP model is trainable.
    rgb2gray_fn = 'luma'  # Function to warp RGB ('color_cdf', 'mlp', 'luma', 'constant_zero').

    def __init__(self, **kwargs):
        super(BilateralGrid3D, self).__init__()
        set_kwargs(self, kwargs)

        # Initialize identity grids.
        init_grids = self.init_identity_grid()
        self.grids = nn.Parameter(init_grids)

        if self.learn_gray:
            self.rgb2gray_fn = Rgb2Gray.MLP
        self.rgb2gray = Rgb2Gray(self.rgb2gray_fn,
                                 color_cdf_spline_coeffs=kwargs.get('color_cdf_spline_coeffs', None))
        if not self.training_rgb2gray_mlp:
            for param in self.rgb2gray.parameters():
                param.requires_grad = False

    def init_identity_grid(self):
        grid = torch.tensor([1., 0, 0, 0, 0, 1., 0, 0, 0, 0, 1., 0, ]).float()
        grid = grid.repeat([self.grid_W * self.grid_Z * self.grid_Y * self.grid_X, 1])
        grid = grid.reshape(self.grid_W, self.grid_Z, self.grid_Y, self.grid_X, -1)
        grid = grid.permute(4, 0, 1, 2, 3)  # (12, grid_W, grid_Z, grid_Y, grid_X)
        return grid

    def forward(self, xyz, rgb):
        """
        :param xyz: (..., 3)
        :param rgb: (..., 3)
        :return: (..., 3, 4)
        """
        sh_ = xyz.shape
        xyz = xyz.reshape(-1, 3)  # flatten (N, 3)
        rgb = rgb.reshape(-1, 3)  # flatten (N, 3)

        bound = 2
        xyz = xyz / bound

        gray = self.rgb2gray(rgb)
        xyzw = torch.cat([xyz, gray], dim=-1)  # (N, 4)

        coef = grid_sample_4d(self.grids, xyzw)
        return coef.reshape(*sh_[:-1], 3, 4)
