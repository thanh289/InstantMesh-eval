from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def resize_chw(img_chw: np.ndarray, size: int, mode: str = "bilinear") -> np.ndarray:
    x = torch.from_numpy(img_chw).float().unsqueeze(0)
    if mode == "nearest":
        x = F.interpolate(x, size=(size, size), mode=mode)
    else:
        x = F.interpolate(x, size=(size, size), mode=mode, align_corners=False)
    return x[0].numpy()


def load_rgba_cv2(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[-1] == 3:
        alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
        img = np.concatenate([img, alpha[..., None]], axis=-1)
    return img[..., [2, 1, 0, 3]]


def rgba_to_rgb_mask(rgba_uint8: np.ndarray, white_bg: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rgba = rgba_uint8.astype(np.float32) / 255.0
    rgb = rgba[..., :3]
    mask = rgba[..., 3:4]
    if white_bg:
        rgb = rgb * mask + (1.0 - mask)
    return rgb.astype(np.float32), mask.astype(np.float32)


def load_depth_file(depth_dir: str, view_name: str) -> np.ndarray:
    npz = os.path.join(depth_dir, f"{view_name}.npz")
    npy = os.path.join(depth_dir, f"{view_name}.npy")
    if os.path.exists(npz):
        f = np.load(npz)
        if "depth" in f:
            arr = f["depth"].astype(np.float32)
        elif "data" in f:
            arr = f["data"].astype(np.float32)
        else:
            raise KeyError(f"No 'depth' or 'data' key in {npz}. Keys={list(f.keys())}")
    elif os.path.exists(npy):
        arr = np.load(npy).astype(np.float32)
    else:
        raise FileNotFoundError(f"Depth not found: {depth_dir}/{view_name}.npz/.npy")

    if arr.ndim == 1:
        side = int(np.sqrt(arr.shape[0]))
        arr = arr.reshape(side, side)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr[0]
    return arr


def depth_to_hw_resized(depth_hw: np.ndarray, size: int) -> np.ndarray:
    x = torch.from_numpy(depth_hw[None, None]).float()
    x = F.interpolate(x, size=(size, size), mode="nearest")
    return x[0, 0].numpy()
