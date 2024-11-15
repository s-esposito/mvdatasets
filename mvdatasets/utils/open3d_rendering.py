import open3d as o3d
import numpy as np

from mvdatasets.utils.raycasting import get_camera_rays


def render_o3d_mesh_normals(camera, o3d_mesh):
    return render_o3d_mesh(camera, o3d_mesh)["normals"]


def render_o3d_mesh_depth(camera, o3d_mesh):
    return render_o3d_mesh(camera, o3d_mesh)["depth"]


def render_o3d_mesh(camera, o3d_mesh):
    """render and open3d mesh with open3d raycasting from camera"""

    # gen rays
    rays_o, rays_d, _ = get_camera_rays(camera)
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    rays = o3d.core.Tensor(
        np.concatenate([rays_o, rays_d], axis=1),
        dtype=o3d.core.Dtype.Float32,
    )

    # setup scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))

    ans = scene.cast_rays(rays)

    hits = np.logical_not(np.isinf(ans["t_hit"].numpy()))
    hits = hits.reshape(camera.height, camera.width)[..., None]

    depth = ans["t_hit"].numpy()
    depth[np.isinf(depth)] = 0.0
    depth = depth.reshape(camera.height, camera.width)[..., None]

    normals = ans["primitive_normals"].numpy()
    normals = (normals + 1.0) * 0.5
    normals = normals.reshape(camera.height, camera.width, 3)
    normals = normals * hits

    return {"hits": hits, "depth": depth, "normals": normals}


def render_o3d_scene(camera, o3d_scene):
    """render and open3d mesh with open3d raycasting from camera"""

    # gen rays
    rays_o, rays_d, _ = get_camera_rays(camera)
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    rays = o3d.core.Tensor(
        np.concatenate([rays_o, rays_d], axis=1),
        dtype=o3d.core.Dtype.Float32,
    )

    ans = o3d_scene.cast_rays(rays)

    hits = np.logical_not(np.isinf(ans["t_hit"].numpy()))
    hits = hits.reshape(camera.height, camera.width)[..., None]

    depth = ans["t_hit"].numpy()
    depth[np.isinf(depth)] = 0.0
    depth = depth.reshape(camera.height, camera.width)[..., None]

    normals = ans["primitive_normals"].numpy()
    normals = (normals + 1.0) * 0.5
    normals = normals.reshape(camera.height, camera.width, 3)
    normals = normals * hits

    geom_ids = ans["geometry_ids"].numpy()
    geom_ids = geom_ids.reshape(camera.height, camera.width, 1)

    return {"hits": hits, "depth": depth, "normals": normals, "geom_ids": geom_ids}
