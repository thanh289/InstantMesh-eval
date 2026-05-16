from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
# from transformers.convert_graph_to_onnx import parser


@dataclass
class EvalConfig:
    data_path: str
    depth_path: Optional[str]
    eval_path: str
    outdir: str = "/workspace/outputs/sf3d_eval"
    csv_name: str = "sf3d_eval_results.csv"
    resume_csv: Optional[str] = None

    max_objects: Optional[int] = None
    object_start: Optional[int] = None
    object_end: Optional[int] = None
    val_size: float = 1.0

    sf3d_model_id: str = "stabilityai/stable-fast-3d"
    sf3d_input_view_index: int = 0
    texture_resolution: int = 512
    remesh_option: str = "none"
    target_vertex_count: int = -1

    fovy: float = 60.0
    cam_radius: float = 1.5
    output_size: int = 512
    depth_render_size: int = 512

    use_rembg: bool = False
    resize_foreground_for_sf3d: bool = True
    foreground_ratio: float = 0.85

    mesh_scale: float = 1.0
    mesh_rot_x_deg: float = 0.0
    mesh_rot_y_deg: float = 0.0
    mesh_rot_z_deg: float = 0.0
    mesh_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    flip_uv_y: bool = True

    # Optional GT mesh folder for mesh metrics.
    # Expected layouts supported by dataset.py:
    #   gt_mesh_path/archive_xxx/object_id.glb
    #   gt_mesh_path/chunk_xxx/object_id.glb
    #   gt_mesh_path/archive_xxx/object_id/mesh.glb
    gt_mesh_path: Optional[str] = None
    mesh_num_samples: int = 100000
    mesh_sample_seed: int = 42
    mesh_fscore_thresholds: tuple[float, ...] = (0.1, 0.2, 0.5)

    save_preview_every: int = 1
    save_mesh: bool = True
    preview_only: bool = False

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    object_list: Optional[str] = None
    allow_missing_object_list: bool = False


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Run SF3D mesh reconstruction + RGB/depth/mesh evaluation.")

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--depth-path", default=None, help="Depth folder. Nếu không cung cấp, depth metrics sẽ bị bỏ qua.")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--outdir", default="/workspace/outputs/sf3d_eval")
    parser.add_argument("--csv-name", default="sf3d_eval_results.csv")
    parser.add_argument("--resume-csv", default=None)

    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--object-start", type=int, default=None)
    parser.add_argument("--object-end", type=int, default=None)
    parser.add_argument("--val-size", type=float, default=1.0)

    parser.add_argument("--sf3d-model-id", default="stabilityai/stable-fast-3d")
    parser.add_argument("--sf3d-input-view-index", type=int, default=0)
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument("--remesh-option", choices=["none", "triangle", "quad"], default="none")
    parser.add_argument("--target-vertex-count", type=int, default=-1)

    parser.add_argument("--fovy", type=float, default=60.0)
    parser.add_argument("--cam-radius", type=float, default=1.5)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--depth-render-size", type=int, default=512)

    parser.add_argument("--use-rembg", action="store_true")
    parser.add_argument("--no-resize-foreground", dest="resize_foreground_for_sf3d", action="store_false")
    parser.set_defaults(resize_foreground_for_sf3d=True)
    parser.add_argument("--foreground-ratio", type=float, default=0.85)

    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--mesh-rot-x-deg", type=float, default=0.0)
    parser.add_argument("--mesh-rot-y-deg", type=float, default=0.0)
    parser.add_argument("--mesh-rot-z-deg", type=float, default=0.0)
    parser.add_argument("--mesh-translation", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--flip-uv-y", dest="flip_uv_y", action="store_true")
    parser.add_argument("--no-flip-uv-y", dest="flip_uv_y", action="store_false")
    parser.set_defaults(flip_uv_y=True)

    parser.add_argument("--gt-mesh-path", default=None, help="Folder containing GT GLB meshes for CD/F-score metrics.")
    parser.add_argument("--mesh-num-samples", type=int, default=100_000, help="Number of surface points sampled per mesh.")
    parser.add_argument("--mesh-sample-seed", type=int, default=42, help="Random seed for mesh surface sampling.")
    parser.add_argument(
        "--mesh-fscore-thresholds",
        type=float,
        nargs="+",
        default=(0.1, 0.2, 0.5),
        help="Distance thresholds for mesh F-score.",
    )

    parser.add_argument("--save-preview-every", type=int, default=1)
    parser.add_argument("--no-save-mesh", dest="save_mesh", action="store_false")
    parser.add_argument("--preview-only", action="store_true", help="Run only the first selected object and save previews.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--object-list", default=None, help="CSV/text file chứa object IDs cần eval. Chỉ chạy những object này.")
    parser.add_argument("--allow-missing-object-list", action="store_true", help="Nếu set, bỏ qua object trong list mà không có data. Mặc định: báo lỗi.")

    args = parser.parse_args()
    cfg = EvalConfig(**vars(args))
    cfg.mesh_fscore_thresholds = tuple(float(x) for x in cfg.mesh_fscore_thresholds)
    cfg.mesh_translation = tuple(float(x) for x in cfg.mesh_translation)
    return cfg