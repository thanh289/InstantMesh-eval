from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .cameras import EVAL_CAMERA_PARAMS, INPUT_VIEW_IDS
from .config import EvalConfig
from .image_io import depth_to_hw_resized, load_depth_file, load_rgba_cv2, resize_chw, rgba_to_rgb_mask
from .sf3d_input import make_sf3d_input_image


def load_object_list(path: str | None) -> set[str] | None:
    """Read a CSV or plain-text file of object IDs.

    CSV: supports columns object_id, archive_name/item_name, ply_path, glb_path.
         object_id is inferred from whichever columns are present.
    Plain text: one object_id per line.

    Each ID is stored in two forms so matching is flexible:
        "archive/item"  and  "item"
    """
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return None

    allowed: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)

        if "," in sample or "object_id" in sample:
            reader = csv.DictReader(f)
            for row in reader:
                oid      = str(row.get("object_id",    "")).strip()
                archive  = str(row.get("archive_name", "")).strip()
                item     = str(row.get("item_name",    "")).strip()
                ply_path = str(row.get("ply_path",     "")).strip()
                glb_path = str(row.get("glb_path",     "")).strip()

                if not oid and archive and item:
                    oid = f"{archive}/{item}"
                if not oid and ply_path:
                    stem   = os.path.splitext(os.path.basename(ply_path))[0]
                    parent = os.path.basename(os.path.dirname(ply_path))
                    oid    = f"{parent}/{stem}" if parent else stem
                if not oid and glb_path:
                    stem   = os.path.splitext(os.path.basename(glb_path))[0]
                    parent = os.path.basename(os.path.dirname(glb_path))
                    oid    = f"{parent}/{stem}" if parent else stem

                if oid:
                    allowed.add(oid)
                    allowed.add(oid.split("/")[-1])
        else:
            for line in f:
                oid = line.strip()
                if oid:
                    allowed.add(oid)
                    allowed.add(oid.split("/")[-1])

    return allowed


def is_allowed_object(object_id: str, allowed: set[str] | None) -> bool:
    if allowed is None:
        return True
    return object_id in allowed or object_id.split("/")[-1] in allowed


class SF3DEvalDataset(Dataset):
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.depth_path = cfg.depth_path
        self.eval_path = cfg.eval_path
        self.gt_mesh_path = cfg.gt_mesh_path
        self.output_size = cfg.output_size
        self.depth_render_size = cfg.depth_render_size

        required_paths = [self.data_path, self.eval_path]
        if self.depth_path is not None:
            required_paths.append(self.depth_path)
        if self.gt_mesh_path is not None:
            required_paths.append(self.gt_mesh_path)

        for required_path in required_paths:
            if not os.path.isdir(required_path):
                raise FileNotFoundError(f"Directory not found: {required_path}")

        if self.depth_path is not None:
            # Scan objects từ depth_path (behavior cũ)
            depth_items = [
                (archive, obj, os.path.join(self.depth_path, archive, obj))
                for archive in sorted(os.listdir(self.depth_path))
                if os.path.isdir(os.path.join(self.depth_path, archive))
                for obj in sorted(os.listdir(os.path.join(self.depth_path, archive)))
                if os.path.isdir(os.path.join(self.depth_path, archive, obj, "depth"))
            ]
        else:
            # Không có depth_path → scan objects từ data_path
            print("[dataset] depth_path not provided — scanning objects from data_path, depth metrics disabled.")
            depth_items = [
                (archive, obj, os.path.join(self.data_path, archive, obj))
                for archive in sorted(os.listdir(self.data_path))
                if os.path.isdir(os.path.join(self.data_path, archive))
                for obj in sorted(os.listdir(os.path.join(self.data_path, archive)))
                if os.path.isdir(os.path.join(self.data_path, archive, obj))
            ]

        n = len(depth_items)
        end = max(0, int(cfg.val_size * n))
        depth_items = depth_items[-end:] if end > 0 else []

        if cfg.object_start is not None or cfg.object_end is not None:
            depth_items = depth_items[cfg.object_start : cfg.object_end]

        if cfg.max_objects is not None:
            depth_items = depth_items[: cfg.max_objects]

        allowed = load_object_list(getattr(cfg, "object_list", None))
        if allowed is not None:
            before = len(depth_items)
            depth_items = [
                (arch, obj, path) for arch, obj, path in depth_items
                if is_allowed_object(f"{arch}/{obj}", allowed)
            ]
            print(f"[dataset] object-list filter: {before} -> {len(depth_items)}")

        self.items = depth_items
        self.eval_index = self._build_eval_index()
        self.gt_mesh_index = self._build_gt_mesh_index()
        print(f"[dataset] objects = {len(self.items)}")

    def _build_eval_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for archive in sorted(os.listdir(self.eval_path)):
            archive_path = os.path.join(self.eval_path, archive)
            if not os.path.isdir(archive_path):
                continue
            for obj in sorted(os.listdir(archive_path)):
                p = os.path.join(archive_path, obj)
                if os.path.isdir(p):
                    index[obj] = p
        return index

    def _build_gt_mesh_index(self) -> dict[str, str]:
        """
        Walk gt_mesh_path recursively and index every .glb file by object name.

        Supported layouts (all resolved to the same flat {item_name -> path} dict):
          gt_mesh_path/archive_xxx/object_id.glb          -> key: object_id
          gt_mesh_path/chunk_xxx/object_id.glb            -> key: object_id
          gt_mesh_path/archive_xxx/object_id/mesh.glb     -> key: object_id  (dir name)
          gt_mesh_path/archive_xxx/object_id/object_id.glb -> key: object_id

        The first match wins; deeper / later paths do NOT overwrite earlier ones so
        that a deterministic walk order (sorted) gives stable results.
        """
        index: dict[str, str] = {}
        if self.gt_mesh_path is None:
            return index

        for root, dirs, files in os.walk(self.gt_mesh_path):
            dirs.sort()  # deterministic traversal order
            for fname in sorted(files):
                if not fname.lower().endswith(".glb"):
                    continue
                full_path = os.path.join(root, fname)
                stem = os.path.splitext(fname)[0]
                # If the file is named "mesh.glb" use the parent directory name as the key,
                # otherwise use the file stem directly.
                item_name = os.path.basename(root) if stem == "mesh" else stem
                if item_name not in index:
                    index[item_name] = full_path

        return index

    def __len__(self) -> int:
        return len(self.items)

    def _find_eval_item_path(self, item_name: str) -> str:
        if item_name in self.eval_index:
            return self.eval_index[item_name]
        raise FileNotFoundError(f"Cannot find {item_name} in EVAL_PATH={self.eval_path}")

    def _find_gt_mesh_path(self, archive_name: str, item_name: str) -> str | None:
        if self.gt_mesh_path is None:
            return None

        if item_name in self.gt_mesh_index:
            return self.gt_mesh_index[item_name]

        raise FileNotFoundError(
            f"Cannot find GT mesh for item={item_name} anywhere under {self.gt_mesh_path}. "
            f"Indexed {len(self.gt_mesh_index)} objects. "
            "Check that the folder contains .glb files and that the object name matches."
        )

    def __getitem__(self, idx: int) -> dict:
        archive_name, item_name, _ = self.items[idx]
        item_path = os.path.join(self.data_path, archive_name, item_name)
        eval_item_path = self._find_eval_item_path(item_name)

        input_view_id = INPUT_VIEW_IDS[self.cfg.sf3d_input_view_index]
        input_view_name = f"{input_view_id:03d}"
        input_rgb_path = os.path.join(item_path, "rgb", f"{input_view_name}.png")
        input_rgba = load_rgba_cv2(input_rgb_path)
        sf3d_input = make_sf3d_input_image(input_rgba, self.cfg)

        input_rgb_preview, input_mask_preview = rgba_to_rgb_mask(input_rgba, white_bg=True)
        input_rgb_preview = resize_chw(input_rgb_preview.transpose(2, 0, 1), self.output_size, mode="bilinear").transpose(1, 2, 0)
        input_mask_preview = resize_chw(input_mask_preview.transpose(2, 0, 1), self.output_size, mode="nearest").transpose(1, 2, 0)

        target_rgbs, target_masks = [], []
        for view_idx, (elev, azim) in enumerate(EVAL_CAMERA_PARAMS):
            view_name = f"{view_idx:03d}"
            rgba = load_rgba_cv2(os.path.join(eval_item_path, "rgb", f"{view_name}.png"))
            rgb, mask = rgba_to_rgb_mask(rgba, white_bg=True)
            rgb = resize_chw(rgb.transpose(2, 0, 1), self.output_size, mode="bilinear").transpose(1, 2, 0)
            mask = resize_chw(mask.transpose(2, 0, 1), self.output_size, mode="nearest").transpose(1, 2, 0)
            target_rgbs.append(rgb)
            target_masks.append(mask)

        # Depth evaluation: chỉ chạy nếu depth_path được cung cấp
        depth_masks, gt_depths = [], []
        if self.depth_path is not None:
            depth_dir = os.path.join(eval_item_path, "depth")
            for view_idx, (elev, azim) in enumerate(EVAL_CAMERA_PARAMS):
                view_name = f"{view_idx:03d}"
                rgba = load_rgba_cv2(os.path.join(eval_item_path, "rgb", f"{view_name}.png"))
                _, mask = rgba_to_rgb_mask(rgba, white_bg=True)
                mask = resize_chw(mask.transpose(2, 0, 1), self.depth_render_size, mode="nearest").transpose(1, 2, 0)
                depth = load_depth_file(depth_dir, view_name)
                depth = depth_to_hw_resized(depth, self.depth_render_size)
                depth_masks.append(mask)
                gt_depths.append(depth[..., None])

        return {
            "archive_name": archive_name,
            "item_name": item_name,
            "object_id": f"{archive_name}/{item_name}",
            "gt_mesh_path": self._find_gt_mesh_path(archive_name, item_name),
            "input_view_name": input_view_name,
            "input_rgb_preview": torch.from_numpy(input_rgb_preview.astype(np.float32)),
            "input_mask_preview": torch.from_numpy(input_mask_preview.astype(np.float32)),
            "sf3d_input": sf3d_input,
            "target_rgbs": torch.from_numpy(np.stack(target_rgbs).astype(np.float32)),
            "target_masks": torch.from_numpy(np.stack(target_masks).astype(np.float32)),
            "target_camera_params": torch.tensor(EVAL_CAMERA_PARAMS, dtype=torch.float32),
            "depth_masks": torch.from_numpy(np.stack(depth_masks).astype(np.float32)) if depth_masks else torch.zeros(0),
            "gt_depths": torch.from_numpy(np.stack(gt_depths).astype(np.float32)) if gt_depths else torch.zeros(0),
            "depth_camera_params": torch.tensor(EVAL_CAMERA_PARAMS, dtype=torch.float32),
        }