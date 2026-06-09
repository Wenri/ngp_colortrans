# ngp_colortrans

NeRF **color transfer** experiments built on [ngp_pl](https://github.com/kwea123/ngp_pl) — Instant-NGP (NeRF only) in pytorch+cuda trained with pytorch-lightning.

This fork reconstructs the scene in **CIELAB color space** and learns, alongside reconstruction, one or more **chroma transfer heads** guided by semantic segmentation: each head adds a per-class learned chroma offset, so the colors of individual semantic regions can be retargeted (and edited live in the GUI) while luminance and geometry stay fixed.

Upstream references:

* [ngp_pl](https://github.com/kwea123/ngp_pl) — the base of this fork
* [Official CUDA implementation](https://github.com/NVlabs/instant-ngp/tree/master)
* [torch-ngp](https://github.com/ashawkey/torch-ngp)

# :rainbow: What's different from ngp_pl

* **Lab pipeline**: input images are converted to CIELAB at load time via their embedded ICC profile (sRGB is assumed, with a warning, if no profile is present — `datasets/color_utils.py`). The network regresses normalized Lab values — tensors named `rgb` throughout the code actually hold Lab (plus extra channels). Renders are converted back to RGB only for metrics (`kornia.lab_to_rgb`) and for saving (PIL LAB→sRGB ICC conversion).
* **16-channel volume rendering**: the CUDA kernels composite `n_ch = 16` channels per ray (compile-time constant in `models/csrc/include/utils.h`, exposed as `vren.get_total_channels()`): 1 luminance + 2 chroma per transfer head (default 2 heads) + 11 semantic class logits.
* **Segmentation-guided transfer heads** (`models/networks.py`): a `seg_net` predicts class logits from position features; each transfer head's `trans_net` produces a chroma offset from auxiliary color features and a learnable per-class code weighted by the segmentation softmax. Head 0 uses a frozen zero code (plain reconstruction); further heads learn the transfer. The code can be overridden at inference (`trans_net_p1` kwarg), which is what the GUI sliders do.
* **Transfer supervision by thin-plate-spline chroma warp**: `assets/transimg.npz` stores `from_points`/`to_points` in the ab-plane; the colmap loader fits a TPS warp from them and applies it to every pixel's chroma to produce the transfer target appended to the rays (Huber loss `trans` in `losses.py`).
* **Semantic supervision**: per-image logits are loaded from `seg/<image>.npy` next to the image folder; classes are ranked by overall probability, the top 10 are kept individually and the remainder pooled into an "other" class, trained with confidence-weighted cross-entropy.
* **Interactive recoloring GUI**: `show_gui.py` renders the transferred output with one slider per semantic class editing the transfer code live.
* The earlier deferred full-image chrominance loss (`--ray_sampling_strategy deferred_images`, `HistLoss`) is kept in the code but currently disabled (its forward raises `NotImplementedError`).

# :computer: Installation

This implementation has **strict** requirements due to dependencies on other libraries; if you encounter an installation problem due to hardware/software mismatch, there is no intention to support different platforms.

## Hardware

* OS: Ubuntu 20.04+
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB (tested with RTX 2080 Ti), CUDA 11.8+
* 32GB RAM (in order to load full size images)

## Software

* Python>=3.10 (the code uses `match` statements), NumPy < 2.0 (uses `np.float_`)
* Python libraries
    * Install `pytorch>=2.1` matching your CUDA version, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121` (required by pytorch-lightning 2.4)
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) (pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`
* Cuda extension: upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (re-run this each time you pull or modify the code under `models/csrc/`, including the channel count `n_ch`)

# :books: Supported Datasets

Color-transfer training currently runs on **Colmap data only** (the other loaders do not provide segmentation): run `colmap` and get a folder `sparse/0` under which there are `cameras.bin`, `images.bin` and `points3D.bin`. [nerf_llff_data](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing), [mipnerf360 data](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and [HDR-NeRF data](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I) are also supported.

Each scene additionally needs:

* `seg/<image_name>.npy` next to the `images/` folder — per-pixel class logits of shape `(C, H, W)` from a semantic segmenter, one file per training image;
* `assets/transimg.npz` (repo-relative) with `from_points`/`to_points` defining the chroma warp; if absent, an identity transfer target is used.

The dataloaders for NSVF, NeRF++ and RTMV data are inherited from upstream ngp_pl but are not wired to the segmentation/transfer training path.

# :key: Training

```bash
python train.py --dataset_name colmap --root_dir <path/to/scene> --exp_name <name>
```

Each step trains on 8192 rays; reconstruction (Lab MSE), chroma transfer (Huber against the TPS-warped target) and segmentation (cross-entropy) losses are optimized jointly. A pretrained checkpoint can be used as a starting point with `--weight_path <path/to/ckpt>`. More options can be found in [opt.py](opt.py).

Outputs are written to `ckpts/<dataset_name>/<exp_name>/` (checkpoints, plus a slimmed copy), `logs/<dataset_name>/<exp_name>/` (TensorBoard) and `results/<dataset_name>/<exp_name>/<epoch>/` (validation renders). Per test view, the following images are saved: `XXX.png` (reconstruction), `XXX_t.png` (transferred colors), `XXX_s.png` (segmentation overlay), `XXX_d.png` (depth) and, at the first validation, `XXX_gt.png`/`XXX_gt_t.png` (ground truth and its warp target). PSNR/SSIM/LPIPS are computed in RGB after Lab conversion.

# :mag_right: Testing

Use `test.ipynb` to generate images from a checkpoint.

GUI usage: run `python show_gui.py` followed by the **same** hyperparameters used in training (`dataset_name`, `root_dir`, etc) and **add the checkpoint path** with `--ckpt_path <path/to/.ckpt>`. The control window exposes one slider per semantic class (`c0`–`c10`) that edits the transfer code of head 1 in real time.

# Acknowledgements

This repository is a research fork of [kwea123/ngp_pl](https://github.com/kwea123/ngp_pl) (MIT license); the volume rendering CUDA extension (`vren`), training infrastructure and benchmarking scripts come from there. Quality/speed benchmarks of the base implementation are reported in the upstream README and [gallery](GALLERY.md).
