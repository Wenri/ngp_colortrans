import logging
from io import BytesIO

import torch
import torchvision
from PIL import Image, ImageCms
from einops import rearrange
import numpy as np
from kornia.color import rgb_to_yuv
import torchvision.transforms.functional as F


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img > limit, 1.055 * img ** (1 / 2.4) - 0.055, 12.92 * img)
    img[img > 1] = 1  # "clamp" tonemapper
    return img


def read_image(img_path, img_wh, blend_a=True):
    log = logging.getLogger(__name__)
    # Read image
    with Image.open(img_path) as image:
        orig_icc = image.info.get('icc_profile')
        # Extract original ICC profile
        with BytesIO(orig_icc) as icc:
            orig_icc = ImageCms.ImageCmsProfile(icc)
        desc = ImageCms.getProfileDescription(orig_icc)

        # Plot image with original ICC profile
        log.debug('Original ICC profile: {}'.format(desc))

        # Create sRGB ICC profile and convert image to sRGB
        srgb_icc = ImageCms.createProfile('sRGB')
        image = ImageCms.profileToProfile(image, orig_icc, srgb_icc)

    w, h = img_wh
    img = F.resize(image, size=[h, w], interpolation=F.InterpolationMode.BICUBIC)
    img = F.to_tensor(img)
    if img.shape[0] == 4:  # blend A to RGB
        if blend_a:
            img = img[:3] * img[-1:] + (1 - img[-1:])
        else:
            img = img[:3] * img[-1:]

    img = rgb_to_yuv(rearrange(img, 'c h w -> 1 c h w')).squeeze(0)
    img = rearrange(img, 'c h w -> (h w) c')

    return img
