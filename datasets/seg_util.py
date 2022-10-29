import logging
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageCms
from einops import rearrange


def read_seg(img_path, img_wh):
    log = logging.getLogger(__name__)
    fn = Path(img_path)
    fp = fn.parent.with_name('seg') / fn.with_suffix('.npy').name
    w, h = img_wh
    seg = np.load(os.fspath(fp))
    seg = torch.from_numpy(seg)
    seg = F.resize(seg, size=(h, w), interpolation=F.InterpolationMode.BILINEAR, antialias=False)
    seg = rearrange(seg, 'c h w -> (h w) c')
    return seg
