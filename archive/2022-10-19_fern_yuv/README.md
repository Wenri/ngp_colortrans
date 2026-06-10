# fern — YUV/deferred-histogram era (2022-10-19)

Archived artifacts of the last experiment run under the early **YUV color
pipeline** with the deferred full-image chrominance loss, before the project
moved to the CIELAB + semantic-segmentation transfer heads. Kept because the
TensorBoard logs are the only quantitative record of that approach and the
checkpoint avoids a ~6.5 h retrain if it is ever revisited.

**Code state:** commit `5d4790d` ("conver color space of input image to sRGB").
The checkpoint contains only `xyz_encoder` / `dir_encoder` / `rgb_net` and is
**not loadable by the current code** — check out that commit (and reinstall
`vren` from its `models/csrc/`, which composited 3 channels, not 16) to use it.

**Scene:** `fern` from nerf_llff_data (public), `--downsample 0.25 --scale 2.0`,
trained on a single GPU (RTX 2080 Ti class).

## Contents

- `epoch=29_slim.ckpt` — weights of the 30-epoch run (slim: regenerable
  occupancy-grid buffers stripped; no optimizer state was saved either way).
- `runs/version_0_baseline_all_images/` — 7-epoch reconstruction baseline,
  `--ray_sampling_strategy all_images --batch_size 4096`.
  Final test/psnr **27.35**, test/ssim **0.71** (PSNR computed in YUV space).
- `runs/version_4_deferred_30ep/` — the full 30-epoch run that produced the
  checkpoint, `--ray_sampling_strategy deferred_images --batch_size 20480`,
  ~6.4 h. Final test/psnr **26.91** (max 27.49), test/ssim **0.72** (max 0.76).
  View with `tensorboard --logdir runs/`.
  (Stored under `runs/` because the repo `.gitignore` excludes any `logs/`.)
- `renders/epoch29_*` — final test views (3 reconstructions + depth maps).
- `renders/deferred_ep{init,0,28}_{pred,gt}.png` — the deferred loss's
  full-image prediction vs target at start, epoch 0 and epoch 28, documenting
  the chroma-statistics matching (init is pure noise; by ep28 the prediction
  is a slightly desaturated, color-shifted match of the target).

Aborted runs (versions 1–3), the non-slim checkpoint (only added regenerable
grid buffers) and the remaining per-epoch renders were discarded.

Binary files in this directory are stored with **git LFS**.
