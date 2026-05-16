from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict

import imageio.v2 as imageio
import lpips
import numpy as np
import pandas as pd
import torch
from sf3d.system import SF3D
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from .config import EvalConfig
from .dataset import SF3DEvalDataset
from .mesh_utils import apply_user_mesh_transform
from .metrics import compute_depth_metrics, compute_mesh_metrics, compute_rgb_metrics, mesh_fscore_key
from .previews import save_depth_preview, save_input_image, save_rgb_preview
from .renderer import render_mesh_pytorch3d


def _object_output_dir(cfg: EvalConfig, object_id: str) -> str:
    parts = [p for p in str(object_id).replace("\\", "/").split("/") if p]
    if not parts:
        parts = ["unknown_object"]
    return os.path.join(cfg.outdir, *parts)


def _mesh_metric_keys(cfg: EvalConfig) -> list[str]:
    return ["cd"] + [mesh_fscore_key(t) for t in cfg.mesh_fscore_thresholds]


def _base_result_row(cfg: EvalConfig, idx: int, object_id: str) -> dict:
    row = {
        "idx": idx,
        "object_id": object_id,
        "psnr": np.nan,
        "ssim": np.nan,
        "lpips": np.nan,
        "abs_diff": np.nan,
        "delta_1": np.nan,
        "time_sec": np.nan,
        "error": "",
    }
    for key in _mesh_metric_keys(cfg):
        row[key] = np.nan
    return row


def run_one_object(sample: dict, model: SF3D, cfg: EvalConfig, idx: int, lpips_metric, ssim_metric) -> dict:
    row = _base_result_row(cfg, idx, str(sample["object_id"]))

    start = time.time()
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if cfg.device == "cuda" else contextlib.nullcontext()
    with torch.no_grad(), autocast_ctx:
        mesh, _ = model.run_image(
            sample["sf3d_input"],
            bake_resolution=cfg.texture_resolution,
            remesh=cfg.remesh_option,
            vertex_count=cfg.target_vertex_count,
        )
    
    # Giải phóng VRAM sau khi SF3D sinh mesh xong, trước khi render
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
 

    pred_rgb, _, _ = render_mesh_pytorch3d(
        mesh,
        sample["target_camera_params"],
        height=cfg.output_size,
        width=cfg.output_size,
        fovy_deg=cfg.fovy,
        dist=cfg.cam_radius,
        device=cfg.device,
        cfg=cfg,
    )
    gt_rgb = sample["target_rgbs"].permute(0, 3, 1, 2).to(cfg.device)
    rgb_m = compute_rgb_metrics(pred_rgb, gt_rgb, lpips_metric, ssim_metric, cfg.device)

    # Depth metrics: chỉ chạy nếu depth_path được cung cấp
    if cfg.depth_path is not None:
        _, pred_depth, pred_alpha_d = render_mesh_pytorch3d(
            mesh,
            sample["depth_camera_params"],
            height=cfg.depth_render_size,
            width=cfg.depth_render_size,
            fovy_deg=cfg.fovy,
            dist=cfg.cam_radius,
            device=cfg.device,
            cfg=cfg,
        )
        gt_depth = sample["gt_depths"].permute(0, 3, 1, 2).to(cfg.device)
        gt_mask = sample["depth_masks"].permute(0, 3, 1, 2).to(cfg.device)
        depth_m = compute_depth_metrics(pred_depth, gt_depth, pred_alpha_d, gt_mask, cfg.device)
    else:
        pred_depth, pred_alpha_d = None, None
        depth_m = {"abs_diff": torch.tensor(float("nan")), "delta_1": torch.tensor(float("nan"))}

    mesh_m: dict[str, float] = {}
    if cfg.gt_mesh_path is not None:
        # Use the same user transform as the renderer. This matters when you set
        # --mesh-scale/--mesh-rot-*/--mesh-translation to make renders line up.
        pred_mesh_for_metric = apply_user_mesh_transform(mesh, cfg)
        mesh_m = compute_mesh_metrics(
            pred_mesh_for_metric,
            sample["gt_mesh_path"],
            num_samples=cfg.mesh_num_samples,
            thresholds=cfg.mesh_fscore_thresholds,
            seed=cfg.mesh_sample_seed,
        )

    row.update(
        {
            "psnr": float(rgb_m["psnr"].detach().cpu()),
            "ssim": float(rgb_m["ssim"].detach().cpu()),
            "lpips": float(rgb_m["lpips"].detach().cpu()),
            "abs_diff": float(depth_m["abs_diff"].detach().cpu()),
            "delta_1": float(depth_m["delta_1"].detach().cpu()),
            "time_sec": time.time() - start,
        }
    )
    for key, value in mesh_m.items():
        row[key] = float(value)

    obj_out = _object_output_dir(cfg, sample["object_id"])
    os.makedirs(obj_out, exist_ok=True)

    input_rgb_preview = sample.get("input_rgb_preview")
    if input_rgb_preview is not None:
        save_input_image(os.path.join(obj_out, f"input_{sample.get('input_view_name', '000')}.png"), input_rgb_preview)

    if cfg.save_mesh:
        try:
            mesh.export(os.path.join(obj_out, "mesh.glb"), include_normals=True)
        except Exception as exc:
            print("Mesh export failed:", repr(exc))

    if cfg.save_preview_every > 0 and (idx % cfg.save_preview_every) == 0:
        preview_input = input_rgb_preview if input_rgb_preview is not None else gt_rgb[0]
        save_rgb_preview(
            os.path.join(obj_out, "preview_rgb.png"),
            preview_input,
            gt_rgb,
            pred_rgb,
            max_views=min(16, len(gt_rgb)),
        )
        if pred_depth is not None:
            save_depth_preview(
                os.path.join(obj_out, "preview_depth.png"),
                preview_input,
                gt_depth,
                pred_depth,
                pred_alpha_d,
                max_views=min(16, len(gt_depth)),
            )

        # Optional: also save the first rendered prediction and GT separately for quick checking.
        imageio.imwrite(
            os.path.join(obj_out, "gt_rgb_first.png"),
            (gt_rgb[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8),
        )
        imageio.imwrite(
            os.path.join(obj_out, "pred_rgb_first.png"),
            (pred_rgb[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8),
        )

    return row


def write_summary(rows: list[dict], cfg: EvalConfig) -> None:
    df = pd.DataFrame(rows)
    ok = df[df["error"].fillna("").eq("")].copy() if len(df) else df

    summary = {
        "num_objects_total": int(len(df)),
        "num_objects_ok": int(len(ok)),
        "config": asdict(cfg),
    }

    for col in ["psnr", "ssim", "lpips", "abs_diff", "delta_1"] + _mesh_metric_keys(cfg):
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


def _complete_rows_for_current_config(prev_df: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    if len(prev_df) == 0:
        return prev_df

    done_mask = prev_df["object_id"].notna() & (~prev_df["object_id"].astype(str).str.strip().eq(""))
    done_mask &= prev_df.get("error", "").fillna("").eq("")

    # If mesh metrics are enabled, do not skip old rows that do not have the new mesh columns.
    # This avoids silently resuming from an old CSV without CD/F-score values.
    if cfg.gt_mesh_path is not None:
        for col in _mesh_metric_keys(cfg):
            if col not in prev_df.columns:
                done_mask &= False
            else:
                done_mask &= prev_df[col].notna()

    return prev_df.loc[done_mask].copy()


def run_eval(cfg: EvalConfig) -> None:
    torch.set_grad_enabled(False)
    os.makedirs(cfg.outdir, exist_ok=True)
    print("Config:")
    print(json.dumps(asdict(cfg), indent=2))
    print("Device:", cfg.device)

    dataset = SF3DEvalDataset(cfg)
    if len(dataset) == 0:
        raise RuntimeError("No objects found. Check data/depth/eval paths and archive structure.")

    print("Loading SF3D...")
    model = SF3D.from_pretrained(cfg.sf3d_model_id, config_name="config.yaml", weight_name="model.safetensors")
    model.to(cfg.device)
    model.eval()

    lpips_metric = lpips.LPIPS(net="vgg").to(cfg.device).eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)
    print("SF3D loaded.")

    csv_out = os.path.join(cfg.outdir, cfg.csv_name)
    resume_csv = cfg.resume_csv or (csv_out if os.path.exists(csv_out) else None)
    if resume_csv is not None and os.path.exists(resume_csv):
        prev_df = pd.read_csv(resume_csv)
        kept_df = _complete_rows_for_current_config(prev_df, cfg)
        done_ids = set(kept_df["object_id"].astype(str)) if len(kept_df) else set()
        rows = kept_df.to_dict(orient="records")
        dropped = len(prev_df) - len(kept_df)
        print(f"Resuming from {resume_csv}: {len(done_ids)} complete objects will be skipped; {dropped} incomplete/old rows will be recomputed.")
    else:
        done_ids = set()
        rows = []

    max_iter = 1 if cfg.preview_only else len(dataset)
    for idx in tqdm(range(max_iter), desc="SF3D eval"):
        try:
            archive_name, item_name, _ = dataset.items[idx]
            object_id = f"{archive_name}/{item_name}"

            if object_id in done_ids:
                print("[SKIP]", object_id)
                continue

            sample = dataset[idx]
            row = run_one_object(sample, model, cfg, idx, lpips_metric, ssim_metric)

        except Exception as exc:
            row = _base_result_row(cfg, idx, object_id)
            row["error"] = repr(exc)
            print("[ERROR]", object_id, row["error"])

        rows.append(row)
        done_ids.add(object_id)
        pd.DataFrame(rows).to_csv(csv_out, index=False)

        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    print("Saved CSV:", csv_out)
    write_summary(rows, cfg)