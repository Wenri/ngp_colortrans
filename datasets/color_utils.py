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


def read_image(img_path, img_wh):
    log = logging.getLogger(__name__)
    # Read image
    with Image.open(img_path) as image:
        orig_icc = image.info.get('icc_profile')

        if orig_icc:
            # Extract original ICC profile
            with BytesIO(orig_icc) as icc:
                orig_icc = ImageCms.ImageCmsProfile(icc)
            desc = ImageCms.getProfileDescription(orig_icc)
            # Plot image with original ICC profile
            log.debug('Original ICC profile: {}'.format(desc))

        else:
            orig_icc = ImageCms.createProfile('sRGB')
            log.warning('No ICC profile found. Assuming sRGB.')

        # Create sRGB ICC profile and convert image to sRGB
        lab_icc = ImageCms.createProfile('LAB', colorTemp=6500)
        lab = ImageCms.profileToProfile(image, orig_icc, lab_icc, outputMode='LAB')

    w, h = img_wh
    lab = np.asarray(lab)
    lab = np.concatenate((lab[..., 0:1], lab.view(np.int8)[..., 1:3]), dtype=np.float_, axis=-1)
    lab = torch.from_numpy(rearrange(lab, 'h w c -> c h w'))
    lab = F.resize(lab, size=(h, w), interpolation=F.InterpolationMode.BICUBIC, antialias=True)
    lab = rearrange(lab, 'c h w -> (h w) c')
    return lab
