from __future__ import annotations

import torch
import trimesh
from pytorch3d.renderer import BlendParams, FoVPerspectiveCameras, MeshRasterizer, PointLights, RasterizationSettings, SoftPhongShader, look_at_view_transform
from pytorch3d.renderer import AmbientLights, HardFlatShader

from .config import EvalConfig
from .mesh_utils import apply_user_mesh_transform, as_camera_param_list, trimesh_to_pytorch3d


def render_mesh_pytorch3d(mesh: trimesh.Trimesh, camera_params, height: int, width: int, fovy_deg: float, dist: float, device: str, cfg: EvalConfig):
    mesh = apply_user_mesh_transform(mesh, cfg)
    p3d_mesh = trimesh_to_pytorch3d(mesh, device=device, flip_uv_y=cfg.flip_uv_y)

    raster_settings = RasterizationSettings(image_size=(height, width), blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    lights = AmbientLights(device=device)
    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
    shader = HardFlatShader (device=device, lights=lights, blend_params=blend_params)

    rgbs, depths, alphas = [], [], []
    for elev, azim in as_camera_param_list(camera_params):
        r_mat, t_vec = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(device=device, R=r_mat, T=t_vec, fov=fovy_deg)
        fragments = rasterizer(p3d_mesh, cameras=cameras)
        image = shader(fragments, p3d_mesh, cameras=cameras)

        alpha = (fragments.pix_to_face[..., :1] >= 0).float()
        rgb = image[..., :3].clamp(0, 1)
        rgb = rgb * alpha + (1.0 - alpha)
        depth = fragments.zbuf[..., :1]
        depth = torch.where(alpha > 0.5, depth, torch.zeros_like(depth))

        rgbs.append(rgb[0].permute(2, 0, 1).contiguous())
        depths.append(depth[0].permute(2, 0, 1).contiguous())
        alphas.append(alpha[0].permute(2, 0, 1).contiguous())

    return torch.stack(rgbs), torch.stack(depths), torch.stack(alphas)
