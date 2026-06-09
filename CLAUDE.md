# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [kwea123/ngp_pl](https://github.com/kwea123/ngp_pl) (Instant-NGP NeRF in PyTorch Lightning) for **semantic-segmentation-guided color transfer** (branch `colortrans`). The scene is reconstructed in **CIELAB color space**; in addition to reconstruction, the network learns chroma "transfer heads" that recolor semantic regions via per-class learned codes, supervised by a thin-plate-spline chroma warp. Tensors and dict keys named `rgb` throughout the code actually hold Lab values plus extra channels — this is the single most important convention in the repo.

## Environment & build

Strict CUDA-only dependency chain (see README for install order): torch 1.11 + CUDA 11.3, `tinycudann`, NVIDIA `apex` (FusedAdam), `torch-scatter`, pytorch-lightning 1.7, then `pip install -r requirements.txt`. Python ≥ 3.10 (`match` statements), NumPy < 2.0 (`np.float_`). Training runs with mixed precision (`precision=16`) on GPU only.

The custom CUDA volume-rendering extension `vren` lives in `models/csrc/` and must be (re)installed after any change to its `.cu`/`.cpp`/`.h` files or after pulling:

```bash
pip install models/csrc/
```

The number of composited channels is a compile-time constant `n_ch = 16` in `models/csrc/include/utils.h`, exposed to Python as `vren.get_total_channels()` (imported as `N_CH` in `models/rendering.py`). Changing the channel budget means editing that constant and reinstalling the extension.

There is no test suite or linter. `misc/differentiable_histogram.py` has a `__main__` self-check.

## Common commands

```bash
# Color-transfer training — colmap scenes only (other loaders lack segmentation)
python train.py --dataset_name colmap --root_dir <path/to/scene> --exp_name <name>

# Start from a pretrained checkpoint (weights only)
python train.py ... --weight_path <path/to/ckpt>

# Validation only
python train.py ... --val_only --ckpt_path <ckpt>

# GUI with live per-class recoloring sliders: same hyperparameters as training, plus checkpoint
python show_gui.py --dataset_name colmap --root_dir <path> ... --ckpt_path <path/to/.ckpt>
```

All options are in `opt.py`. Outputs land in `ckpts/<dataset_name>/<exp_name>/` (checkpoints + `_slim` copy), `logs/<dataset_name>/<exp_name>/` (TensorBoard), `results/<dataset_name>/<exp_name>/<epoch>/` (per view: `XXX.png` reconstruction, `XXX_t.png` transferred, `XXX_s.png` segmentation overlay, `XXX_d.png` depth, `XXX_gt[_t].png` at first validation).

## Data requirements (colmap scenes)

- `sparse/0/{cameras,images,points3D}.bin` from COLMAP; images under `images/`.
- `seg/<image_name>.npy` — per-pixel class logits `(C, H, W)` from a semantic segmenter, one per image (`datasets/seg_util.py`).
- `assets/transimg.npz` (repo-relative path, loaded by `ColmapDataset`) — `from_points`/`to_points` in the ab-plane defining the TPS chroma warp used as transfer supervision. If missing, the loader logs an exception and uses the identity (no warp).
- Images are converted to Lab via their embedded ICC profile; sRGB is assumed (warning logged) if absent.

Training only works with `--dataset_name colmap`: `NeRFSystem.configure_optimizers` calls `train_dataset.sort_sem()`, and `NeRFLoss` expects `batch['seg']` — both exist only in `ColmapDataset`. The NSVF/NeRF/NeRF++/RTMV loaders are inherited from upstream and not wired to this path.

## Architecture

**Channel layout** (the contract among model, CUDA kernels, losses, and savers): each ray/sample carries `N_CH = 16` channels = `[L, a0, b0, a1, b1, seg_logits × 11]`. `NGP.n_total_color_ch = 2 * n_trans_head + 1` (default `n_trans_head=2` → 5); the remaining 11 are semantic classes. Head 0 is reconstruction chroma, head 1 transferred chroma.

**Model — `models/networks.py`.** `NGPBase` holds the multi-cascade density/occupancy grid machinery (`mark_invisible_cells`, `update_density_grid`, morton-coded bitfield consumed by the CUDA ray marcher). `NGP`: hash-grid `xyz_encoder` (32-dim features; sigma = softplus of feature 0), SH `dir_encoder`, `rgb_net` (dir + position features → 1 luma + 2 chroma + 13 auxiliary features), view-independent `seg_net` (position features → 11 class logits), and `trans_net` (aux features + segmentation-softmax-weighted per-class code → 2-channel chroma offset added to base chroma, tanh). Per-head codes are `trans_net_p{i}` — head 0 a frozen zero buffer, head ≥ 1 learnable parameters — and can be overridden per render call via kwargs (`trans_net_p1=...`), which is how `show_gui.py` does live recoloring. Output activations: sigmoid on L, tanh on all chroma, raw seg logits. `models/nerf_helpers.py` has a vanilla positional-encoding NeRF alternative (commented swap in `NeRFSystem.__init__`).

**Rendering — `models/rendering.py` + `models/custom_functions.py` + `vren`.** `render()` intersects rays with the scene AABB, then `__render_rays_train` (custom-autograd `RayMarcher` + `VolumeRenderer` over the whole batch) or `__render_rays_test` (iterative marching of alive rays, no grad). All 16 channels are composited by the CUDA kernels. Tensor kwargs passed to `render()` are repeated per sample via `rays_a` before reaching the model.

**Losses — `losses.py`.** `NeRFLoss(n_color_ch)` returns a dict summed in `training_step`: `rgb` (MSE on Lab channels 0–2), `trans` (Huber, weight 1e-1, on transferred chroma vs the TPS-warped target stored as ray channels 3–4), `seg` (confidence-weighted cross-entropy, weight 1e-1: dataset-wide class ranking `sem_ind` from `sort_sem()` keeps the top 10 classes and pools the rest into "other"; only pixels with target confidence ≥ 0.5 contribute), `opacity` (entropy regularizer), optional `distortion` (CUDA Mip-NeRF-360 loss, `--distortion_loss_w`).

**Data — `datasets/`.** `ColmapDataset._read_imgbuf` produces 5-channel rays: Lab from `read_image` (L ∈ [0,255], ab as int8) plus TPS-warped ab, normalized by `(255, 128, 128, 128, 128)` → L ∈ [0,1], chroma ∈ (−1,1). `BaseDataset._getitem_random` implements `--ray_sampling_strategy`; batches carry `rgb` (5ch rays), `seg` (logits), indices. All rays/segs are preloaded into memory; train epochs are hardcoded to 1000 steps.

**Training orchestration — `train.py`.** `NeRFSystem` (LightningModule). Dataloaders use `num_workers=0, batch_size=None` (the dataset itself returns full batches). Occupancy grid refreshes every 16 steps. Metrics: predictions/GT are scaled by `(100, 128, 128)` and converted with `kornia.lab_to_rgb` before PSNR/SSIM/LPIPS (kornia expects L ∈ [0,100]). Saving uses PIL `LAB` mode with scale `(255, 128, 128)` and an ICC LAB→sRGB conversion — note the two different L scales.

## Gotchas

- The deferred full-image path is **disabled**: `--ray_sampling_strategy deferred_images` still exists in `opt.py`/`base.py` and `train.py:deferred_step` still calls `HistLoss`, but `HistLoss.forward` raises `NotImplementedError` immediately. It (and the differentiable histograms in `misc/differentiable_histogram.py`) are remnants of the earlier YUV/histogram approach kept for reference.
- `NGP.forward` asserts all outputs are finite — non-finite values crash training intentionally.
- `show_gui.py` pops a one-time matplotlib debug window on the first render (`_DEBUG_TRIG`) and hardcodes `_N_CLS = 11`.
- `save_image_trans` saves both the reconstruction and the `_t` transferred variant for whatever rays it is given (including GT).
