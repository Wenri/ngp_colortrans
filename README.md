# ngp_colortrans

NeRF **color transfer** experiments built on [ngp_pl](https://github.com/kwea123/ngp_pl) — Instant-NGP (NeRF only) in pytorch+cuda trained with pytorch-lightning.

This fork modifies the original reconstruction pipeline to operate in **YUV color space** and adds a deferred full-image loss on the chrominance channels, so that the color statistics of the rendered scene can be driven toward a target while luminance/geometry remain supervised per ray.

Upstream references:

* [ngp_pl](https://github.com/kwea123/ngp_pl) — the base of this fork
* [Official CUDA implementation](https://github.com/NVlabs/instant-ngp/tree/master)
* [torch-ngp](https://github.com/ashawkey/torch-ngp)

# :rainbow: What's different from ngp_pl

* **YUV pipeline**: input images are converted from their embedded ICC profile to sRGB, then to YUV at load time (`datasets/color_utils.py`). The network regresses YUV directly — tensors named `rgb` throughout the code actually hold YUV. Renders are converted back to RGB only for saving and for SSIM/LPIPS.
* **Separate luma/chroma heads**: the NGP color output uses a sigmoid activation for Y and tanh for the signed UV channels (`models/networks.py`).
* **Deferred color loss**: the hidden ray sampling strategy `deferred_images` samples all rays of a batch from a single image and additionally returns the whole image. Each step, the full image is rendered without gradients, the current batch's differentiable ray predictions are scattered into it, and a chrominance loss (`HistLoss` in `losses.py`: UV channel means + UV spatial gradients) is applied on the composite — full-image color statistics with per-batch memory cost.
* **Differentiable color tools** in `misc/`: Gaussian / multivariate-Gaussian soft histograms and an RGB↔Lab formulation, used as alternative color-statistics losses.
* `--weight_path` can load a pretrained scene checkpoint (weights only) as the starting point of an experiment.

# :computer: Installation

This implementation has **strict** requirements due to dependencies on other libraries; if you encounter an installation problem due to hardware/software mismatch, there is no intention to support different platforms.

## Hardware

* OS: Ubuntu 20.04+
* NVIDIA GPU with Compute Compatibility >= 75 and memory > 6GB (tested with RTX 2080 Ti), CUDA 11.3+
* 32GB RAM (in order to load full size images)

## Software

* Python>=3.10 (the code uses `match` statements; installation via [anaconda](https://www.anaconda.com/distribution/) is recommended)
* Python libraries
    * Install `pytorch>=1.11.0` with the CUDA version matching your setup
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`
* Cuda extension: upgrade `pip` to >= 22.1 and run `pip install models/csrc/` (re-run this each time you pull or modify the code under `models/csrc/`)

# :books: Supported Datasets

1.  NSVF data: download preprocessed datasets (`Synthetic_NeRF`, `Synthetic_NSVF`, `BlendedMVS`, `TanksAndTemples`) from [NSVF](https://github.com/facebookresearch/NSVF#dataset). **Do not change the folder names** since there is some hard-coded fix in the dataloader.

2.  NeRF++ data: download from [here](https://github.com/Kai-46/nerfplusplus#data).

3.  Colmap data: for custom data, run `colmap` and get a folder `sparse/0` under which there are `cameras.bin`, `images.bin` and `points3D.bin`. [nerf_llff_data](https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing), [mipnerf360 data](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and [HDR-NeRF data](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I) are also supported.

4.  RTMV data: download from [here](http://www.cs.umd.edu/~mmeshry/projects/rtmv/) and run `python misc/prepare_rtmv.py <path/to/RTMV>` to convert the hdr images into ldr images for training.

**Note**: images must carry an embedded ICC profile — image loading converts from that profile to sRGB and fails if it is missing.

# :key: Training

Quickstart:

```bash
python train.py --root_dir <path/to/lego> --exp_name Lego
```

It will train the Lego scene for 30k steps (each step with 8192 rays), and perform one testing at the end. Add `--no_save_test` to skip saving test images (which is slow). Note that reported training/testing PSNR is computed in YUV space.

Color transfer training uses the deferred full-image loss, typically starting from a pretrained scene:

```bash
python train.py --root_dir <path/to/scene> --exp_name <name> \
    --ray_sampling_strategy deferred_images --weight_path <path/to/pretrained.ckpt>
```

More options can be found in [opt.py](opt.py). Reference invocations for the public datasets are under `benchmarking/`.

Outputs are written to `ckpts/<dataset_name>/<exp_name>/` (checkpoints, plus a slimmed copy), `logs/<dataset_name>/<exp_name>/` (TensorBoard) and `results/<dataset_name>/<exp_name>/<epoch>/` (validation renders; the deferred mode also dumps `deferred_pred.png`/`deferred_gt.png` there for inspection).

# :mag_right: Testing

Use `test.ipynb` to generate images from a checkpoint.

GUI usage: run `python show_gui.py` followed by the **same** hyperparameters used in training (`dataset_name`, `root_dir`, etc) and **add the checkpoint path** with `--ckpt_path <path/to/.ckpt>`.

# Acknowledgements

This repository is a research fork of [kwea123/ngp_pl](https://github.com/kwea123/ngp_pl) (MIT license); the volume rendering CUDA extension (`vren`), training infrastructure and benchmarking scripts come from there. Quality/speed benchmarks of the base implementation are reported in the upstream README and [gallery](GALLERY.md).
