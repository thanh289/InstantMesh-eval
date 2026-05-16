from __future__ import annotations

from typing import Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy_hwc_rgb(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().float().numpy()
    else:
        arr = np.asarray(img, dtype=np.float32)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        rgb = arr[..., :3]
        alpha = arr[..., 3:4]
        arr = rgb * alpha + (1.0 - alpha)

    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _resize_hwc_rgb(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x[0].permute(1, 2, 0).cpu().numpy()


def save_input_image(path: str, input_rgb) -> None:
    img = _to_numpy_hwc_rgb(input_rgb)
    imageio.imwrite(path, (img * 255).astype(np.uint8))


def save_rgb_preview(path: str, input_rgb, gt_rgb: torch.Tensor, pred_rgb: torch.Tensor, max_views: int = 16) -> None:
    gt = gt_rgb[:max_views].detach().cpu().numpy().transpose(0, 2, 3, 1)
    pr = pred_rgb[:max_views].detach().cpu().numpy().transpose(0, 2, 3, 1)

    if len(gt) == 0:
        raise ValueError("gt_rgb is empty")

    tile_h, tile_w = gt[0].shape[:2]
    inp = _resize_hwc_rgb(_to_numpy_hwc_rgb(input_rgb), (tile_h, tile_w))
    blank = np.ones_like(inp, dtype=np.float32)

    n = min(len(gt), len(pr), max_views)
    row_input = np.concatenate([inp] + [blank] * (n - 1), axis=1)
    row_gt = np.concatenate(list(gt[:n]), axis=1)
    row_pr = np.concatenate(list(pr[:n]), axis=1)
    canvas = np.concatenate([row_input, row_gt, row_pr], axis=0)

    imageio.imwrite(path, (np.clip(canvas, 0, 1) * 255).astype(np.uint8))


def depth_vis(depth: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
    d = depth.copy()
    if alpha is not None:
        d[alpha <= 0.1] = np.nan
    valid = np.isfinite(d) & (d > 0)
    out = np.zeros((*d.shape, 3), dtype=np.float32)
    if valid.sum() == 0:
        return out
    lo, hi = np.percentile(d[valid], [2, 98])
    x = (d - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0, 1)
    cmap = plt.get_cmap("viridis")
    out = cmap(x)[..., :3].astype(np.float32)
    out[~valid] = 1.0
    return out


def save_depth_preview(path: str, input_rgb, gt_depth: torch.Tensor, pred_depth: torch.Tensor, pred_alpha: torch.Tensor, max_views: int = 16) -> None:
    gt = gt_depth[:max_views, 0].detach().cpu().numpy()
    pr = pred_depth[:max_views, 0].detach().cpu().numpy()
    al = pred_alpha[:max_views, 0].detach().cpu().numpy()

    gt_imgs = [depth_vis(gt[i]) for i in range(len(gt))]
    pr_imgs = [depth_vis(pr[i], al[i]) for i in range(len(pr))]

    if len(gt_imgs) == 0:
        raise ValueError("gt_depth is empty")

    tile_h, tile_w = gt_imgs[0].shape[:2]
    inp = _resize_hwc_rgb(_to_numpy_hwc_rgb(input_rgb), (tile_h, tile_w))
    blank = np.ones_like(inp, dtype=np.float32)

    n = min(len(gt_imgs), len(pr_imgs), max_views)
    row_input = np.concatenate([inp] + [blank] * (n - 1), axis=1)
    row_gt = np.concatenate(gt_imgs[:n], axis=1)
    row_pr = np.concatenate(pr_imgs[:n], axis=1)
    canvas = np.concatenate([row_input, row_gt, row_pr], axis=0)

    imageio.imwrite(path, (np.clip(canvas, 0, 1) * 255).astype(np.uint8))
