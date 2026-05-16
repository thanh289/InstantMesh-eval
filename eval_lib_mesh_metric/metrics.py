from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


def compute_rgb_metrics(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, lpips_metric, ssim_metric, device: str) -> dict[str, torch.Tensor]:
    pred_rgb = pred_rgb.to(device).float().clamp(0, 1)
    gt_rgb = gt_rgb.to(device).float().clamp(0, 1)

    mse = F.mse_loss(pred_rgb, gt_rgb)
    psnr = -10.0 * torch.log10(mse + 1e-8)
    ssim = ssim_metric(pred_rgb, gt_rgb)

    pred_256 = F.interpolate(pred_rgb * 2 - 1, size=(256, 256), mode="bilinear", align_corners=False)
    gt_256 = F.interpolate(gt_rgb * 2 - 1, size=(256, 256), mode="bilinear", align_corners=False)
    lp = lpips_metric(pred_256, gt_256).mean()
    return {"psnr": psnr.detach(), "ssim": ssim.detach(), "lpips": lp.detach()}


def compute_depth_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor, pred_alpha: torch.Tensor, gt_mask: torch.Tensor, device: str, min_valid: int = 10) -> dict[str, torch.Tensor]:
    pred_depth = pred_depth.to(device).float()
    gt_depth = gt_depth.to(device).float()
    pred_alpha = pred_alpha.to(device).float()
    gt_mask = gt_mask.to(device).float()

    abs_list, delta_list = [], []
    for v in range(pred_depth.shape[0]):
        mask = (pred_alpha[v, 0] > 0.1) & (gt_mask[v, 0] > 0.01) & (gt_depth[v, 0] > 0.01) & (pred_depth[v, 0] > 0.0)
        if mask.sum() < min_valid:
            continue
        pred = pred_depth[v, 0][mask]
        gt = gt_depth[v, 0][mask]
        abs_diff = torch.abs(pred - gt).mean()
        thresh = torch.max(pred / (gt + 1e-8), gt / (pred + 1e-8))
        delta_1 = (thresh < 1.25).float().mean()
        abs_list.append(abs_diff)
        delta_list.append(delta_1)

    if not abs_list:
        return {"abs_diff": torch.tensor(float("nan"), device=device), "delta_1": torch.tensor(float("nan"), device=device)}
    return {"abs_diff": torch.stack(abs_list).mean().detach(), "delta_1": torch.stack(delta_list).mean().detach()}


def mesh_fscore_key(threshold: float) -> str:
    text = f"{float(threshold):g}".replace("-", "m").replace(".", "_")
    return f"fscore_{text}"


@contextmanager
def _numpy_seed(seed: int | None):
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _as_single_trimesh(asset) -> trimesh.Trimesh:
    if isinstance(asset, trimesh.Trimesh):
        mesh = asset.copy()
    elif isinstance(asset, trimesh.Scene):
        mesh = None
        try:
            mesh = asset.to_geometry()
        except Exception:
            pass
        if mesh is None or not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            dumped = asset.dump(concatenate=True)
            if isinstance(dumped, list):
                meshes = [m for m in dumped if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0 and len(m.faces) > 0]
                if not meshes:
                    raise ValueError("Scene contains no valid mesh geometry.")
                mesh = trimesh.util.concatenate(meshes)
            elif isinstance(dumped, trimesh.Trimesh):
                mesh = dumped
            else:
                raise ValueError(f"Could not convert Scene to Trimesh: {type(dumped)}")
    else:
        raise TypeError(f"Unsupported mesh type: {type(asset)}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Empty mesh; cannot compute mesh metrics.")
    mesh.remove_unreferenced_vertices()
    return mesh


def load_mesh_for_metrics(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GT mesh file not found: {path}")
    asset = trimesh.load(str(path), force="scene", process=False)
    return _as_single_trimesh(asset)


def _sample_surface_points(mesh: trimesh.Trimesh, count: int, seed: int | None = None) -> np.ndarray:
    mesh = _as_single_trimesh(mesh)
    if count <= 0:
        raise ValueError(f"mesh_num_samples must be positive, got {count}")
    if mesh.area <= 0:
        raise ValueError("Mesh has non-positive surface area; cannot sample surface points.")
    with _numpy_seed(seed):
        points, _ = trimesh.sample.sample_surface(mesh, int(count))
    return np.asarray(points, dtype=np.float32)


def _nearest_distances(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise ImportError("scipy is required for fast mesh metrics. Install it with: pip install scipy") from exc

    tree = cKDTree(dst_points)
    try:
        distances, _ = tree.query(src_points, k=1, workers=-1)
    except TypeError:
        distances, _ = tree.query(src_points, k=1)
    return np.asarray(distances, dtype=np.float32)


def compute_mesh_metrics(
    pred_mesh: trimesh.Trimesh,
    gt_mesh_path: str | Path,
    num_samples: int = 100_000,
    thresholds: Iterable[float] = (0.1, 0.2, 0.5),
    seed: int | None = 42,
) -> dict[str, float]:
    """
    Compute mesh CD and F-score by uniformly sampling points on the predicted and GT surfaces.

    CD here is symmetric mean nearest-neighbor Euclidean distance:
        mean(pred -> gt) + mean(gt -> pred)

    F-score@tau uses:
        precision = fraction of pred samples within tau of GT
        recall    = fraction of GT samples within tau of pred
        fscore    = 2 * precision * recall / (precision + recall)
    """
    gt_mesh = load_mesh_for_metrics(gt_mesh_path)

    pred_points = _sample_surface_points(pred_mesh, num_samples, seed=seed)
    gt_points = _sample_surface_points(gt_mesh, num_samples, seed=None if seed is None else seed + 1)

    pred_to_gt = _nearest_distances(pred_points, gt_points)
    gt_to_pred = _nearest_distances(gt_points, pred_points)

    metrics: dict[str, float] = {
        "cd": float(pred_to_gt.mean() + gt_to_pred.mean()),
    }

    for threshold in thresholds:
        threshold = float(threshold)
        precision = float((pred_to_gt < threshold).mean())
        recall = float((gt_to_pred < threshold).mean())
        fscore = 0.0 if (precision + recall) <= 0 else (2.0 * precision * recall / (precision + recall))
        metrics[mesh_fscore_key(threshold)] = float(fscore)

    return metrics
