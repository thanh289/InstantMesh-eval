#!/usr/bin/env python3
"""
Evaluation script for InstantMesh (instant-mesh-large / FlexiCubes variant).

Metrics computed:
  - RGB : PSNR, SSIM, LPIPS
  - Mesh: Chamfer Distance (CD), F-score @ configurable thresholds

Evaluation camera layout:
  16 views — EVAL_CAMERA_PARAMS from cameras.py
  fov=60°, cam_radius=1.5

Input camera layout (Zero123++ convention):
  6 views — elevations [20, -10, 20, -10, 20, -10]°,
             azimuths  [30, 90, 150, 210, 270, 330]° relative to query image
  fov=30°, cam_radius=4.0

Usage
-----
python run_instantmesh_eval.py \
    --data-path   /path/to/dataset \
    --eval-path   /path/to/eval_renders \
    --outdir      /path/to/outputs \
    --model-path  /path/to/instant_mesh_large.ckpt \
    --config      configs/instant-mesh-large.yaml \
    [--gt-mesh-path /path/to/gt_meshes] \
    [--object-list  object_ids.csv] \
    [--max-objects  100] \
    [--resume-csv   /path/to/previous_results.csv]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import imageio.v2 as imageio
import lpips as lpips_lib
import numpy as np
import pandas as pd
import torch
import trimesh
from omegaconf import OmegaConf
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import v2
from tqdm import tqdm

# ── InstantMesh imports  ─────────────────────────
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras

from eval_lib_mesh_metric.cameras import EVAL_CAMERA_PARAMS
from eval_lib_mesh_metric.image_io import load_rgba_cv2, resize_chw, rgba_to_rgb_mask
from eval_lib_mesh_metric.metrics import (
    compute_mesh_metrics,
    compute_rgb_metrics,
    mesh_fscore_key,
)
from eval_lib_mesh_metric.renderer import render_mesh_pytorch3d
from eval_lib_mesh_metric.dataset import load_object_list, is_allowed_object
from eval_lib_mesh_metric.previews import save_input_image, save_rgb_preview

# ─────────────────────────────────────────────────────────────────────────────
# Camera constants
# ─────────────────────────────────────────────────────────────────────────────

# Zero123++ input camera layout (matches ValidationData in objaverse.py and run.py)
INPUT_FOV   = 30.0
INPUT_DIST  = 4.0

EVAL_FOV    = 60.0
EVAL_DIST   = 1.5
EVAL_SIZE   = 512          # rendered image resolution


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    data_path: str
    eval_path: str
    outdir: str = "outputs/instantmesh_eval"
    csv_name: str = "instantmesh_eval_results.csv"
    resume_csv: Optional[str] = None

    config: str = "configs/instant-mesh-large.yaml"
    model_path: Optional[str] = None          # local ckpt; if None, load from HF hub

    texture_resolution: int = 512
    output_size: int = EVAL_SIZE

    gt_mesh_path: Optional[str] = None
    mesh_num_samples: int = 100_000
    mesh_sample_seed: int = 42
    mesh_fscore_thresholds: tuple = (0.1, 0.2, 0.5)

    max_objects: Optional[int] = None
    object_start: Optional[int] = None
    object_end: Optional[int] = None
    val_size: float = 1.0
    object_list: Optional[str] = None

    save_mesh: bool = True
    save_preview_every: int = 1
    preview_only: bool = False

    # mesh transform (passed straight to SF3D renderer helpers)
    mesh_scale: float = 1.0
    mesh_rot_x_deg: float = 0.0
    mesh_rot_y_deg: float = 0.0
    mesh_rot_z_deg: float = 0.0
    mesh_translation: tuple = (0.0, 0.0, 0.0)
    flip_uv_y: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate InstantMesh: RGB + mesh metrics.")

    parser.add_argument("--data-path",  required=True)
    parser.add_argument("--eval-path",  required=True)
    parser.add_argument("--outdir",     default="outputs/instantmesh_eval")
    parser.add_argument("--csv-name",   default="instantmesh_eval_results.csv")
    parser.add_argument("--resume-csv", default=None)

    parser.add_argument("--config",     default="configs/instant-mesh-large.yaml")
    parser.add_argument("--model-path", default=None)

    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument("--output-size",        type=int, default=EVAL_SIZE)

    parser.add_argument("--gt-mesh-path",           default=None)
    parser.add_argument("--mesh-num-samples",        type=int,   default=100_000)
    parser.add_argument("--mesh-sample-seed",        type=int,   default=42)
    parser.add_argument("--mesh-fscore-thresholds",  type=float, nargs="+", default=[0.1, 0.2, 0.5])

    parser.add_argument("--max-objects",   type=int,   default=None)
    parser.add_argument("--object-start",  type=int,   default=None)
    parser.add_argument("--object-end",    type=int,   default=None)
    parser.add_argument("--val-size",      type=float, default=1.0)
    parser.add_argument("--object-list",   default=None)

    parser.add_argument("--no-save-mesh",        dest="save_mesh",      action="store_false")
    parser.add_argument("--save-preview-every",  type=int, default=1)
    parser.add_argument("--preview-only",        action="store_true")

    parser.add_argument("--mesh-scale",       type=float, default=1.0)
    parser.add_argument("--mesh-rot-x-deg",   type=float, default=0.0)
    parser.add_argument("--mesh-rot-y-deg",   type=float, default=0.0)
    parser.add_argument("--mesh-rot-z-deg",   type=float, default=0.0)
    parser.add_argument("--mesh-translation", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--no-flip-uv-y",     dest="flip_uv_y", action="store_false")
    parser.set_defaults(flip_uv_y=True, save_mesh=True)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    cfg = EvalConfig(**{k: v for k, v in vars(args).items()})
    cfg.mesh_fscore_thresholds = tuple(float(x) for x in cfg.mesh_fscore_thresholds)
    cfg.mesh_translation = tuple(float(x) for x in cfg.mesh_translation)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scanning  (mirrors SF3D dataset.py logic, without depth)
# ─────────────────────────────────────────────────────────────────────────────



def scan_objects(cfg: EvalConfig):
    """Return list of (archive_name, item_name, item_data_path)."""
    items = [
        (archive, obj, os.path.join(cfg.data_path, archive, obj))
        for archive in sorted(os.listdir(cfg.data_path))
        if os.path.isdir(os.path.join(cfg.data_path, archive))
        for obj in sorted(os.listdir(os.path.join(cfg.data_path, archive)))
        if os.path.isdir(os.path.join(cfg.data_path, archive, obj))
    ]

    n = len(items)
    end = max(0, int(cfg.val_size * n))
    items = items[-end:] if end > 0 else []

    if cfg.object_start is not None or cfg.object_end is not None:
        items = items[cfg.object_start : cfg.object_end]

    if cfg.max_objects is not None:
        items = items[: cfg.max_objects]

    allowed = load_object_list(cfg.object_list)
    if allowed is not None:
        before = len(items)
        items = [
            (arch, obj, path) for arch, obj, path in items
            if is_allowed_object(f"{arch}/{obj}", allowed)
        ]
        print(f"[dataset] object-list filter: {before} -> {len(items)}")

    print(f"[dataset] objects = {len(items)}")
    return items


def build_eval_index(eval_path: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for archive in sorted(os.listdir(eval_path)):
        archive_path = os.path.join(eval_path, archive)
        if not os.path.isdir(archive_path):
            continue
        for obj in sorted(os.listdir(archive_path)):
            p = os.path.join(archive_path, obj)
            if os.path.isdir(p):
                index[obj] = p
    return index


def build_gt_mesh_index(gt_mesh_path: Optional[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    if gt_mesh_path is None:
        return index
    for root, dirs, files in os.walk(gt_mesh_path):
        dirs.sort()
        for fname in sorted(files):
            if not fname.lower().endswith(".glb"):
                continue
            full_path = os.path.join(root, fname)
            stem = os.path.splitext(fname)[0]
            item_name = os.path.basename(root) if stem == "mesh" else stem
            if item_name not in index:
                index[item_name] = full_path
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: EvalConfig):
    om_cfg = OmegaConf.load(cfg.config)
    model = instantiate_from_config(om_cfg.model_config)

    if cfg.model_path and os.path.exists(cfg.model_path):
        ckpt_path = cfg.model_path
    else:
        from huggingface_hub import hf_hub_download
        config_name = os.path.basename(cfg.config).replace(".yaml", "")
        ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename=f"{config_name.replace('-', '_')}.ckpt",
            repo_type="model",
        )

    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(cfg.device)
    model.init_flexicubes_geometry(cfg.device, fovy=INPUT_FOV)
    model = model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Input image preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_input_views(item_path: str, output_size: int) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load the 6 Zero123++ input views from the dataset folder.

    Expected layout:  item_path/rgb/000.png … 007.png  (or similar).
    View 000 is treated as the query image (front-facing).
    The 6 input views are views 0,1,2,3,4,5 mapped to azimuths 30°,90°,...,330°.

    Returns
    -------
    images_t : (1, 6, 3, 320, 320) float32 tensor in [0,1]  — model input
    query_rgb : (H, W, 3) float32 numpy array in [0,1]      — for preview
    """
    rgb_dir = os.path.join(item_path, "rgb")

    # We need 6 input views. Use the first 6 available sorted PNGs.
    all_pngs = sorted(
        f for f in os.listdir(rgb_dir) if f.endswith(".png")
    )
    if len(all_pngs) < 6:
        raise FileNotFoundError(
            f"Need at least 6 input view images in {rgb_dir}, found {len(all_pngs)}"
        )

    view_files = all_pngs[:6]   # views 000–005 → azimuths 30°–330°

    imgs = []
    for fname in view_files:
        rgba = load_rgba_cv2(os.path.join(rgb_dir, fname))
        rgb, mask = rgba_to_rgb_mask(rgba, white_bg=True)          # (H,W,3), (H,W,1)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()       # (3,H,W)
        t = v2.functional.resize(t.unsqueeze(0), 320,
                                  interpolation=3, antialias=True).clamp(0, 1).squeeze(0)
        imgs.append(t)

    images_t = torch.stack(imgs, dim=0).unsqueeze(0)               # (1,6,3,320,320)

    # Query / preview image: view 000 resized to output_size
    rgba0 = load_rgba_cv2(os.path.join(rgb_dir, all_pngs[0]))
    query_rgb, _ = rgba_to_rgb_mask(rgba0, white_bg=True)
    query_rgb = resize_chw(
        query_rgb.transpose(2, 0, 1), output_size, mode="bilinear"
    ).transpose(1, 2, 0)

    return images_t, query_rgb


def load_eval_views(eval_item_path: str, output_size: int):
    """
    Load the 16 ground-truth eval views (EVAL_CAMERA_PARAMS layout).

    Returns
    -------
    target_rgbs  : (16, 3, H, W) float32 tensor in [0,1]
    target_masks : (16, 1, H, W) float32 tensor in [0,1]
    """
    target_rgbs, target_masks = [], []
    for view_idx in range(len(EVAL_CAMERA_PARAMS)):
        view_name = f"{view_idx:03d}"
        rgba = load_rgba_cv2(os.path.join(eval_item_path, "rgb", f"{view_name}.png"))
        rgb, mask = rgba_to_rgb_mask(rgba, white_bg=True)
        rgb  = resize_chw(rgb.transpose(2, 0, 1),  output_size, mode="bilinear").transpose(1, 2, 0)
        mask = resize_chw(mask.transpose(2, 0, 1), output_size, mode="nearest").transpose(1, 2, 0)
        target_rgbs.append(rgb)
        target_masks.append(mask)

    target_rgbs  = torch.from_numpy(np.stack(target_rgbs).astype(np.float32))   # (16,H,W,3)
    target_masks = torch.from_numpy(np.stack(target_masks).astype(np.float32))  # (16,H,W,1)
    return target_rgbs, target_masks


# ─────────────────────────────────────────────────────────────────────────────
# Per-object evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _mesh_metric_keys(cfg: EvalConfig) -> list[str]:
    return ["cd"] + [mesh_fscore_key(t) for t in cfg.mesh_fscore_thresholds]


def _base_row(cfg: EvalConfig, idx: int, object_id: str) -> dict:
    row = {
        "idx": idx,
        "object_id": object_id,
        "psnr": float("nan"),
        "ssim": float("nan"),
        "lpips": float("nan"),
        "time_sec": float("nan"),
        "error": "",
    }
    for key in _mesh_metric_keys(cfg):
        row[key] = float("nan")
    return row


def numpy_to_trimesh(vertices: np.ndarray, faces: np.ndarray,
                     vertex_colors: np.ndarray) -> trimesh.Trimesh:
    """Convert InstantMesh extract_mesh() output to a trimesh.Trimesh."""
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )
    return mesh


def run_one_object(
    idx: int,
    archive_name: str,
    item_name: str,
    item_path: str,
    eval_item_path: str,
    gt_mesh_path_for_item: Optional[str],
    model,
    input_cameras: torch.Tensor,
    cfg: EvalConfig,
    lpips_metric,
    ssim_metric,
) -> dict:
    object_id = f"{archive_name}/{item_name}"
    row = _base_row(cfg, idx, object_id)
    start = time.time()

    # ── 1. Load inputs ────────────────────────────────────────────────────────
    images_t, query_rgb = load_input_views(item_path, cfg.output_size)
    images_t = images_t.to(cfg.device)

    # ── 2. Run InstantMesh: planes → mesh ────────────────────────────────────
    # autocast only covers forward_planes (encoder + transformer).
    # extract_mesh() runs FlexiCubes geometry prediction which has scatter/index
    # ops that require float32 — mixing dtypes causes index_add_() dtype errors.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if cfg.device == "cuda" else contextlib.nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            planes = model.forward_planes(images_t, input_cameras)
        # Back to float32 for FlexiCubes mesh extraction
        planes = planes.float()
        vertices_np, faces_np, vertex_colors_np = model.extract_mesh(
            planes,
            use_texture_map=False,
            texture_resolution=cfg.texture_resolution,
        )

    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    # ── 3. Build trimesh ──────────────────────────────────────────────────────
    mesh = numpy_to_trimesh(vertices_np, faces_np, vertex_colors_np)

    # ── 4. Render predicted mesh at eval cameras (PyTorch3D) ─────────────────
    eval_camera_params = torch.tensor(EVAL_CAMERA_PARAMS, dtype=torch.float32)
    pred_rgb, _, _ = render_mesh_pytorch3d(
        mesh,
        eval_camera_params,
        height=cfg.output_size,
        width=cfg.output_size,
        fovy_deg=EVAL_FOV,
        dist=EVAL_DIST,
        device=cfg.device,
        cfg=cfg,
    )   # (16, 3, H, W)

    # ── 5. Load GT eval views ─────────────────────────────────────────────────
    target_rgbs, _ = load_eval_views(eval_item_path, cfg.output_size)
    gt_rgb = target_rgbs.permute(0, 3, 1, 2).to(cfg.device)   # (16,3,H,W)

    # ── 6. RGB metrics ────────────────────────────────────────────────────────
    rgb_m = compute_rgb_metrics(pred_rgb, gt_rgb, lpips_metric, ssim_metric, cfg.device)

    # ── 7. Mesh metrics ───────────────────────────────────────────────────────
    mesh_m: dict[str, float] = {}
    if cfg.gt_mesh_path is not None and gt_mesh_path_for_item is not None:
        from eval_lib_mesh_metric.mesh_utils import apply_user_mesh_transform
        pred_mesh_for_metric = apply_user_mesh_transform(mesh, cfg)
        mesh_m = compute_mesh_metrics(
            pred_mesh_for_metric,
            gt_mesh_path_for_item,
            num_samples=cfg.mesh_num_samples,
            thresholds=cfg.mesh_fscore_thresholds,
            seed=cfg.mesh_sample_seed,
        )

    # ── 8. Collect results ────────────────────────────────────────────────────
    row.update({
        "psnr":     float(rgb_m["psnr"].detach().cpu()),
        "ssim":     float(rgb_m["ssim"].detach().cpu()),
        "lpips":    float(rgb_m["lpips"].detach().cpu()),
        "time_sec": time.time() - start,
    })
    for key, value in mesh_m.items():
        row[key] = float(value)

    # ── 9. Save outputs ───────────────────────────────────────────────────────
    obj_out = os.path.join(cfg.outdir, archive_name, item_name)
    os.makedirs(obj_out, exist_ok=True)

    save_input_image(os.path.join(obj_out, "input_000.png"), query_rgb)

    if cfg.save_mesh:
        try:
            mesh.export(os.path.join(obj_out, "mesh.glb"))
        except Exception as exc:
            print("Mesh export failed:", repr(exc))

    if cfg.save_preview_every > 0 and (idx % cfg.save_preview_every) == 0:
        save_rgb_preview(
            os.path.join(obj_out, "preview_rgb.png"),
            query_rgb,
            gt_rgb,
            pred_rgb,
            max_views=min(16, len(gt_rgb)),
        )
        imageio.imwrite(
            os.path.join(obj_out, "gt_rgb_first.png"),
            (gt_rgb[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8),
        )
        imageio.imwrite(
            os.path.join(obj_out, "pred_rgb_first.png"),
            (pred_rgb[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8),
        )

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(rows: list[dict], cfg: EvalConfig) -> None:
    df = pd.DataFrame(rows)
    ok = df[df["error"].fillna("").eq("")].copy() if len(df) else df

    summary = {
        "num_objects_total": int(len(df)),
        "num_objects_ok":    int(len(ok)),
        "config": asdict(cfg),
    }
    for col in ["psnr", "ssim", "lpips"] + _mesh_metric_keys(cfg):
        if len(ok) and col in ok.columns:
            value = ok[col].mean()
            summary[col] = None if pd.isna(value) else float(value)
        else:
            summary[col] = None

    summary_path = os.path.join(cfg.outdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("Saved summary:", summary_path)


def _complete_rows(prev_df: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    if len(prev_df) == 0:
        return prev_df
    done_mask = (
        prev_df["object_id"].notna()
        & (~prev_df["object_id"].astype(str).str.strip().eq(""))
        & prev_df.get("error", "").fillna("").eq("")
    )
    if cfg.gt_mesh_path is not None:
        for col in _mesh_metric_keys(cfg):
            if col not in prev_df.columns:
                done_mask &= False
            else:
                done_mask &= prev_df[col].notna()
    return prev_df.loc[done_mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Main eval loop
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(cfg: EvalConfig) -> None:
    torch.set_grad_enabled(False)
    os.makedirs(cfg.outdir, exist_ok=True)
    print("Config:", json.dumps(asdict(cfg), indent=2))
    print("Device:", cfg.device)

    # Scan objects
    items = scan_objects(cfg)
    if not items:
        raise RuntimeError("No objects found. Check --data-path and folder structure.")

    # Build lookup indices
    eval_index    = build_eval_index(cfg.eval_path)
    gt_mesh_index = build_gt_mesh_index(cfg.gt_mesh_path)

    # Load model
    print("Loading InstantMesh model …")
    model = load_model(cfg)
    print("Model loaded.")

    # Pre-compute input cameras once (shared across all objects)
    input_cameras = get_zero123plus_input_cameras(
        batch_size=1, radius=INPUT_DIST
    ).to(cfg.device)   # (1, 6, 16)

    # Metrics
    lpips_metric = lpips_lib.LPIPS(net="vgg").to(cfg.device).eval()
    ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)

    # Resume
    csv_out    = os.path.join(cfg.outdir, cfg.csv_name)
    resume_csv = cfg.resume_csv or (csv_out if os.path.exists(csv_out) else None)
    if resume_csv and os.path.exists(resume_csv):
        prev_df  = pd.read_csv(resume_csv)
        kept_df  = _complete_rows(prev_df, cfg)
        done_ids = set(kept_df["object_id"].astype(str)) if len(kept_df) else set()
        rows     = kept_df.to_dict(orient="records")
        dropped  = len(prev_df) - len(kept_df)
        print(f"Resuming from {resume_csv}: {len(done_ids)} done, {dropped} will be recomputed.")
    else:
        done_ids = set()
        rows     = []

    max_iter = 1 if cfg.preview_only else len(items)

    for idx in tqdm(range(max_iter), desc="InstantMesh eval"):
        archive_name, item_name, item_path = items[idx]
        object_id = f"{archive_name}/{item_name}"

        if object_id in done_ids:
            print("[SKIP]", object_id)
            continue

        # Resolve eval path
        if item_name not in eval_index:
            row = _base_row(cfg, idx, object_id)
            row["error"] = f"eval path not found for {item_name}"
            print("[ERROR]", row["error"])
            rows.append(row)
            done_ids.add(object_id)
            pd.DataFrame(rows).to_csv(csv_out, index=False)
            continue

        eval_item_path = eval_index[item_name]

        # Resolve GT mesh path (optional)
        gt_mesh_path_for_item: Optional[str] = None
        if cfg.gt_mesh_path is not None:
            if item_name in gt_mesh_index:
                gt_mesh_path_for_item = gt_mesh_index[item_name]
            else:
                row = _base_row(cfg, idx, object_id)
                row["error"] = f"GT mesh not found for {item_name}"
                print("[ERROR]", row["error"])
                rows.append(row)
                done_ids.add(object_id)
                pd.DataFrame(rows).to_csv(csv_out, index=False)
                continue

        try:
            row = run_one_object(
                idx=idx,
                archive_name=archive_name,
                item_name=item_name,
                item_path=item_path,
                eval_item_path=eval_item_path,
                gt_mesh_path_for_item=gt_mesh_path_for_item,
                model=model,
                input_cameras=input_cameras,
                cfg=cfg,
                lpips_metric=lpips_metric,
                ssim_metric=ssim_metric,
            )
        except Exception as exc:
            row = _base_row(cfg, idx, object_id)
            row["error"] = repr(exc)
            print("[ERROR]", object_id, row["error"])

        rows.append(row)
        done_ids.add(object_id)
        pd.DataFrame(rows).to_csv(csv_out, index=False)

        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    print("Saved CSV:", csv_out)
    write_summary(rows, cfg)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = parse_args()
    run_eval(cfg)