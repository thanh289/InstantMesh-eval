from __future__ import annotations

import numpy as np
from PIL import Image

from sf3d.utils import remove_background, resize_foreground

from .config import EvalConfig


def make_sf3d_input_image(rgba_uint8: np.ndarray, cfg: EvalConfig) -> Image.Image:
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    if cfg.use_rembg:
        img = remove_background(img)
    if cfg.resize_foreground_for_sf3d:
        img = resize_foreground(img, cfg.foreground_ratio, out_size=(512, 512))
    return img
