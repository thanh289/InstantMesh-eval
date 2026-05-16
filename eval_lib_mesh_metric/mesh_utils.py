from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import trimesh
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes

from .config import EvalConfig


def rotation_matrix_np(rx: float = 0, ry: float = 0, rz: float = 0) -> np.ndarray:
    rx, ry, rz = map(math.radians, [rx, ry, rz])
    rx_m = np.array([[1, 0, 0, 0], [0, math.cos(rx), -math.sin(rx), 0], [0, math.sin(rx), math.cos(rx), 0], [0, 0, 0, 1]], dtype=np.float32)
    ry_m = np.array([[math.cos(ry), 0, math.sin(ry), 0], [0, 1, 0, 0], [-math.sin(ry), 0, math.cos(ry), 0], [0, 0, 0, 1]], dtype=np.float32)
    rz_m = np.array([[math.cos(rz), -math.sin(rz), 0, 0], [math.sin(rz), math.cos(rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def apply_user_mesh_transform(mesh: trimesh.Trimesh, cfg: EvalConfig) -> trimesh.Trimesh:
    mesh = mesh.copy()
    trans_m = np.eye(4, dtype=np.float32)
    trans_m[:3, 3] = np.array(cfg.mesh_translation, dtype=np.float32)
    scale_m = np.eye(4, dtype=np.float32)
    scale_m[:3, :3] *= float(cfg.mesh_scale)
    rot_m = rotation_matrix_np(cfg.mesh_rot_x_deg, cfg.mesh_rot_y_deg, cfg.mesh_rot_z_deg)
    mesh.apply_transform(trans_m @ rot_m @ scale_m)
    return mesh


def get_texture_image(mesh: trimesh.Trimesh):
    material = getattr(mesh.visual, "material", None)
    if material is None:
        return None
    tex_img = getattr(material, "baseColorTexture", None)
    if tex_img is None:
        tex_img = getattr(material, "image", None)
    return tex_img


def trimesh_to_pytorch3d(mesh: trimesh.Trimesh, device: str, flip_uv_y: bool = True) -> Meshes:
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Empty mesh; cannot render.")

    verts = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device)
    faces = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int64), device=device)
    texture = None

    has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
    tex_img = get_texture_image(mesh)
    if has_uv and tex_img is not None:
        try:
            uvs_np = np.asarray(mesh.visual.uv, dtype=np.float32)
            if flip_uv_y:
                uvs_np = uvs_np.copy()
                uvs_np[:, 1] = 1.0 - uvs_np[:, 1]

            tex_np = np.asarray(tex_img.convert("RGB"), dtype=np.float32) / 255.0
            verts_uvs = torch.as_tensor(uvs_np, dtype=torch.float32, device=device)[None]
            faces_uvs = faces[None]
            maps = torch.as_tensor(tex_np, dtype=torch.float32, device=device)[None]
            texture = TexturesUV(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
        except Exception as exc:
            print("UV texture fallback triggered:", repr(exc))
            texture = None

    if texture is None:
        try:
            vc = np.asarray(mesh.visual.to_color().vertex_colors[:, :3], dtype=np.float32) / 255.0
            if vc.shape[0] == verts.shape[0]:
                vertex_colors = torch.as_tensor(vc, dtype=torch.float32, device=device)
                texture = TexturesVertex(verts_features=[vertex_colors])
        except Exception as exc:
            print("Vertex color fallback failed:", repr(exc))
            texture = None

    if texture is None:
        vertex_colors = torch.ones_like(verts, dtype=torch.float32, device=device) * 0.7
        texture = TexturesVertex(verts_features=[vertex_colors])

    return Meshes(verts=[verts], faces=[faces], textures=texture)


def as_camera_param_list(camera_params: Iterable) -> list[tuple[float, float]]:
    if torch.is_tensor(camera_params):
        arr = camera_params.detach().cpu().numpy()
    else:
        arr = np.asarray(camera_params, dtype=np.float32)
    arr = arr.reshape(-1, 2)
    return [(float(e), float(a)) for e, a in arr]
