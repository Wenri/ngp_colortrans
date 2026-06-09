# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [kwea123/ngp_pl](https://github.com/kwea123/ngp_pl) (Instant-NGP NeRF in PyTorch Lightning) modified for **color transfer** experiments (branch `colortrans`). The fork's central change: the entire pipeline operates in **YUV color space**, and a "deferred" full-image rendering path applies color-statistics losses on the UV channels.

## Environment & build

Strict CUDA-only dependency chain (see README for install order): PyTorch + CUDA, `tinycudann`, NVIDIA `apex` (FusedAdam), `torch-scatter`, then `pip install -r requirements.txt`. Code uses `match` statements and was run under Python 3.10 (README's 3.8 claim is stale upstream text). Training runs with mixed precision (`precision=16`) on GPU only.

The custom CUDA volume-rendering extension `vren` lives in `models/csrc/` and must be (re)installed after any change to its `.cu`/`.cpp` files or after pulling:

```bash
pip install models/csrc/
```

There is no test suite or linter. `misc/differentiable_histogram.py` has a `__main__` self-check (`python misc/differentiable_histogram.py`).

## Common commands

```bash
# Train (dataset_name: nerf|nsvf|colmap|nerfpp|rtmv, default nsvf)
python train.py --root_dir <path/to/scene> --exp_name <name>

# Color-transfer training (enables the deferred full-image HistLoss path)
python train.py --root_dir <path> --exp_name <name> --ray_sampling_strategy deferred_images

# Validation only
python train.py --root_dir <path> --exp_name <name> --val_only --ckpt_path <ckpt>

# GUI viewer: same hyperparameters as training, plus checkpoint
python show_gui.py --root_dir <path> ... --ckpt_path <path/to/.ckpt>
```

All options are in `opt.py`. Reference training invocations for public datasets are in `benchmarking/*.sh`. `test.ipynb` generates images from a checkpoint.

Outputs land in `ckpts/<dataset_name>/<exp_name>/` (checkpoints, plus a `_slim` copy at the end), `logs/<dataset_name>/<exp_name>/` (TensorBoard), and `results/<dataset_name>/<exp_name>/<epoch>/` (validation renders).

## Architecture

**Training orchestration — `train.py`.** `NeRFSystem` (LightningModule) owns the model, losses, and metrics. Camera poses/ray directions are registered as buffers in `configure_optimizers`; dataloaders use `num_workers=0, batch_size=None` because the dataset's `__getitem__` returns a full ray batch itself. Train epochs are hardcoded to 1000 steps (`BaseDataset.__len__`). The occupancy grid is refreshed every 16 steps via `model.update_density_grid`.

**Model — `models/networks.py`.** `NGPBase` holds the multi-cascade density/occupancy grid machinery (`mark_invisible_cells`, `update_density_grid`, morton-coded bitfield consumed by the CUDA ray marcher). `NGP` adds the tcnn hash-grid encoder, spherical-harmonics direction encoder, and MLP heads. `models/nerf_helpers.py` contains a vanilla positional-encoding NeRF implementing the same `NGPBase` interface — swappable via the commented line in `NeRFSystem.__init__`.

**Rendering — `models/rendering.py` + `models/custom_functions.py` + `vren`.** `render()` intersects rays with the scene AABB, then dispatches to `__render_rays_train` (custom-autograd `RayMarcher` + `VolumeRenderer` over the whole batch) or `__render_rays_test` (iterative marching of alive rays, no grad). `custom_functions.py` wraps the `vren` CUDA kernels (ray marching, compositing, distortion loss) in `torch.autograd.Function`s. Background color is composited onto `results['rgb']` based on `exp_step_factor` (0 → synthetic → ones; else zeros, or random with `--random_bg`).

**Data — `datasets/`.** `dataset_dict` in `__init__.py` maps `--dataset_name` to loaders that read poses/intrinsics per format. All training rays are preloaded into `self.rays` of shape `(N_images, h*w, channels)`. `BaseDataset._getitem_random` implements the `--ray_sampling_strategy` choices; `deferred_images` samples pixels from ONE image and additionally returns that whole image under key `img`.

## Color-transfer specifics (critical fork knowledge)

- **Everything named `rgb` actually holds YUV.** `datasets/color_utils.py:read_image` converts each image from its embedded ICC profile to sRGB, then to YUV via `kornia.color.rgb_to_yuv`. The network regresses YUV; `NeRFLoss` MSE is on YUV; validation PSNR is computed in YUV. Convert back with `yuv_to_rgb` before saving images or computing SSIM/LPIPS (see `NeRFSystem.save_image` / `validation_step`).
- **Input images must have an embedded ICC profile** — `read_image` crashes if `icc_profile` is absent.
- **Output activations**: with `rgb_act=None` (Python `None`, the default in `train.py` unless `--use_exposure`), `NGP.forward` applies sigmoid to the Y channel and tanh to UV (UV is signed). The string `'None'` instead selects the HDR-NeRF log-radiance/tonemapper path — `None` and `'None'` are distinct modes.
- **Deferred color-transfer path** (`--ray_sampling_strategy deferred_images`): when `img` is in the batch, `NeRFSystem.deferred_step` renders the *entire* image under `no_grad`, scatters the current batch's differentiable ray predictions into it, and applies `HistLoss` against the GT image — so gradients flow only through the sampled rays while the loss sees full-image statistics. It also dumps `deferred_pred.png`/`deferred_gt.png` into the val dir on the first batch or on non-finite outputs (then asserts).
- **`HistLoss`** (`losses.py`): currently L1 on UV channel means + L1 on `kornia.filters.spatial_gradient` of UV. Differentiable-histogram variants (`GaussianHistogram`, `MultivariateGaussianHistogram` in `misc/differentiable_histogram.py`, with KL divergence) are wired up but commented out — they are the experiment's alternates, not dead code.
- `misc/rgb_lab_formulation_pytorch.py` provides differentiable RGB↔Lab conversion for the same purpose.
