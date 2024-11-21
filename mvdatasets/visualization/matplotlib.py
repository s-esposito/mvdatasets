import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from itertools import product, combinations
from mvdatasets.utils.raycasting import (
    get_rays_per_points_2d_screen,
)
from mvdatasets.camera import Camera
from mvdatasets.geometry.common import get_mask_points_out_image
from mvdatasets.geometry.primitives import BoundingBox, BoundingSphere
from mvdatasets.utils.printing import print_error


TRANSPARENT = True
BBOX_INCHES = "tight"
PAD_INCHES = 0
DPI = 300
COLORBAR_FRACTION = 0.04625
SCALE_MULTIPLIER = 0.1

# from mvdatasets.scenes.camera import Camera
# import math

# Use in a notebook with:

# import ipywidgets as widgets
# from IPython.display import display
# @widgets.interact(azimuth_deg=(0, 360))
# def f(azimuth_deg=5):
#     plot_fn(..., azimuth_deg=azimuth_deg, ...)


# def _scene_radius(poses):
#     """
#     compute scene radius from list of poses

#     Args:
#         poses (list): list of numpy (4, 4) poses

#     Returns:
#         scene_radius (float): scene radius
#     """
#     # get all camera centers
#     camera_centers = np.stack(poses, 0)[:, :3, 3]
#     camera_distances_from_origin = np.linalg.norm(camera_centers, axis=1)
#     scene_radius = np.max(camera_distances_from_origin)
#     # scene_radius = max(np.max(camera_distances_from_origin) * 0.75, 1.0)
#     return scene_radius


def _draw_3d_init(
    ax,
    scene_radius=1,
    elevation_deg=60,
    azimuth_deg=30,
    up="z"
):
    if scene_radius < 1.0:
        lim = 1.0
    else:
        lim = scene_radius
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([max(-1, -lim), lim])

    ax.set_xlabel("X")
    ax.set_ylabel("Y") if up == "z" else ax.set_ylabel("Z")
    ax.set_zlabel("Z") if up == "z" else ax.set_zlabel("Y")

    # axis equal
    ax.set_aspect("equal")
    ax.view_init(elevation_deg, azimuth_deg)


def _draw_bounding_cube(ax: plt.Axes, scale=1.0):
    # draw bounding cube
    r = [-scale, scale]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="black")


def _draw_rays(
    ax,
    rays_o: np.ndarray,
    rays_d: np.ndarray,
    rgbs=None,
    masks=None,
    max_nr_rays=None,
    ray_lenght=None,
    up="z",
    scene_radius=1.0,
):
    if rays_o is None or rays_d is None:
        return

    # subsample
    if max_nr_rays is not None:
        if max_nr_rays < rays_o.shape[0]:
            idx = np.random.permutation(rays_o.shape[0])[:max_nr_rays]
            rays_o = rays_o[idx]
            rays_d = rays_d[idx]
            if rgbs is not None:
                rgbs = rgbs[idx]
            if masks is not None:
                masks = masks[idx]

    if ray_lenght is None:
        ray_lenght = 2 * scene_radius

    # draw rays
    for i, (ray_o, ray_d) in enumerate(zip(rays_o, rays_d)):
        start_point = ray_o
        end_point = ray_o + ray_d * ray_lenght
        color = rgbs[i] if rgbs is not None else "blue"
        alpha = 0.75
        if masks is not None:
            mask = masks[i]
            if mask < 0.5:
                alpha = 0.5
        # plot line segment
        ax.plot(
            [start_point[0], end_point[0]],
            (
                [start_point[1], end_point[1]]
                if up == "z"
                else [start_point[2], end_point[2]]
            ),
            (
                [start_point[2], end_point[2]]
                if up == "z"
                else [start_point[1], end_point[1]]
            ),
            color=color,
            alpha=0.3 * float(alpha),
        )


def _draw_point_cloud(
    ax,
    points_3d: np.ndarray,
    size=None,
    color=None,
    marker=None,
    label=None,
    max_nr_points=None,
    up="z",
    scene_radius=1.0,
):
    if points_3d is None:
        return

    # subsample
    if max_nr_points is not None:
        if max_nr_points < points_3d.shape[0]:
            idx = np.random.permutation(points_3d.shape[0])[:max_nr_points]
            points_3d = points_3d[idx]

    scale = scene_radius * SCALE_MULTIPLIER

    if color is None:
        # random matplotlib color
        color = "black"
    if size is None:
        size = 10
    if marker is None:
        marker = "o"

    # draw points
    if up == "z":
        ax.scatter(
            points_3d[:, 0],
            points_3d[:, 1],
            points_3d[:, 2],
            s=scale * size,
            color=color,
            marker=marker,
            label=label,
        )
    else:  # up = "y"
        ax.scatter(
            points_3d[:, 0],
            points_3d[:, 2],
            points_3d[:, 1],
            s=scale * size,
            color=color,
            marker=marker,
            label=label,
        )


def _draw_frame(
    ax,
    pose,
    idx=0,
    up="z",
    scene_radius=1.0
):
    if pose is None:
        return

    scale = scene_radius * SCALE_MULTIPLIER

    # get axis directions (normalized)
    x_dir = pose[:3, 0]
    x_dir /= np.linalg.norm(x_dir)
    y_dir = pose[:3, 1]
    y_dir /= np.linalg.norm(y_dir)
    z_dir = pose[:3, 2]
    z_dir /= np.linalg.norm(z_dir)

    # frame center
    pos = pose[:3, 3]

    # draw bb frame
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        x_dir[0],
        x_dir[1] if up == "z" else x_dir[2],
        x_dir[2] if up == "z" else x_dir[1],
        length=scale,
        color="r",
    )
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        y_dir[0],
        y_dir[1] if up == "z" else y_dir[2],
        y_dir[2] if up == "z" else y_dir[1],
        length=scale,
        color="g",
    )
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        z_dir[0],
        z_dir[1] if up == "z" else z_dir[2],
        z_dir[2] if up == "z" else z_dir[1],
        length=scale,
        color="b",
    )
    eps = 0.2 * scale
    ax.text(
        pos[0] + eps,  # x
        pos[1] + eps if up == "z" else pos[2] + eps,  # y
        pos[2] + eps if up == "z" else pos[1] + eps,  # z
        str(idx),
    )


def _draw_cartesian_axis(
    ax,
    up="z",
    scene_radius=1.0
):
    _draw_frame(ax, np.eye(4), idx="w", up=up, scene_radius=scene_radius)


def _draw_bounding_box(
    ax: plt.Axes,
    bb: BoundingBox = None,
    idx: int = 0,
    up: str = "z",
    scene_radius: float = 1.0,
    draw_frame: bool = False
):
    if bb is None:
        return

    scale = scene_radius * SCALE_MULTIPLIER

    # draw bounding box
    segments_indices = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ]
    )

    # get bb pose
    pose = bb.get_pose()
    pose = pose.cpu().numpy()

    # get vertices and pairs of vertices for plotting
    vertices = bb.get_vertices(in_world_space=True)
    vertices = vertices.cpu().numpy()

    vertices_pairs = vertices[segments_indices]

    if bb.color is not None:
        color = bb.color
    else:
        color = "black"

    # visualize min, max vertices
    min_vertex = vertices[0]
    max_vertex = vertices[7]
    ax.scatter(
        min_vertex[0],
        min_vertex[1],
        min_vertex[2],
        s=scale * 50,
        color=color,
        marker="o",
    )
    ax.scatter(
        max_vertex[0],
        max_vertex[1],
        max_vertex[2],
        s=scale * 50,
        color=color,
        marker="o",
    )

    # plot line segments
    for pair in vertices_pairs:
        ax.plot3D(
            *zip(
                pair[0] if up == "z" else pair[0][[0, 2, 1]],
                pair[1] if up == "z" else pair[1][[0, 2, 1]],
            ),
            color=color,
            linewidth=bb.line_width,
            alpha=0.2
        )

    if bb.label is not None:
        label = bb.label
    else:
        label = idx

    # draw bb frame
    if draw_frame:
        _draw_frame(ax, pose, idx=label, up=up, scene_radius=scene_radius)


def _draw_bounding_boxes(
    ax: plt.Axes,
    bounding_boxes: list[BoundingBox] = [],
    up="z",
    scene_radius=1.0,
    draw_frame=False
):
    if bounding_boxes is None:
        return

    # draw bounding boxes
    for i, bb in enumerate(bounding_boxes):
        _draw_bounding_box(
            ax, bb, idx=i, up=up, scene_radius=scene_radius, draw_frame=draw_frame
        )


def _draw_bounding_sphere(
    ax, sphere, idx=0, up="z", scene_radius=1.0, draw_frame=False
):
    if sphere is None:
        return

    # draw sphere at origin
    radius = sphere.get_radius()
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v) * radius
    y = np.sin(u) * np.sin(v) * radius
    z = np.cos(v) * radius
    ax.plot_wireframe(x, y, z, color="black", alpha=0.1)

    if sphere.label is not None:
        label = sphere.label
    else:
        label = idx

    # get bb pose
    pose = sphere.get_pose()

    # draw bb frame
    if draw_frame:
        _draw_frame(ax, pose, idx=label, up=up, scene_radius=scene_radius)


def _draw_bounding_spheres(
    ax,
    bounding_spheres: list[BoundingSphere] = [],
    up="z",
    scene_radius=1.0,
    draw_frame=False
):
    if bounding_spheres is None:
        return

    # draw bounding spheres
    for i, sphere in enumerate(bounding_spheres):
        _draw_bounding_sphere(
            ax=ax,
            sphere=sphere,
            idx=i,
            up=up,
            scene_radius=scene_radius,
            draw_frame=draw_frame
        )


def _draw_image_plane(
    ax,
    camera: Camera,
    up="z",
    scene_radius=1.0
):
    if camera is None:
        return

    scale = scene_radius * SCALE_MULTIPLIER

    # get image plane corner points in 3D
    # from screen coordinates
    corner_points_2d_screen = np.array(
        [[0, 0], [camera.width, 0], [0, camera.height], [camera.width, camera.height]]
    )

    _, corner_points_d = get_rays_per_points_2d_screen(
        torch.from_numpy(camera.get_pose()).float(),
        torch.from_numpy(camera.get_intrinsics_inv()).float(),
        torch.from_numpy(corner_points_2d_screen).float(),
    )
    corner_points_d = corner_points_d.cpu().numpy()

    corner_points_3d_world = camera.get_center() + corner_points_d * scale

    for i, j in combinations(range(4), 2):
        if up == "z":
            ax.plot3D(
                *zip(corner_points_3d_world[i], corner_points_3d_world[j]),
                color="black",
                linewidth=1.0,
                alpha=0.5
            )
        else:
            ax.plot3D(
                *zip(
                    corner_points_3d_world[:, [0, 2, 1]][i],
                    corner_points_3d_world[:, [0, 2, 1]][j],
                ),
                color="black",
                linewidth=1.0,
                alpha=0.5
            )


def _draw_contraction_spheres(ax: plt.Axes):

    # draw sphere at origin
    radius = 1.0
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v) * radius
    y = np.sin(u) * np.sin(v) * radius
    z = np.cos(v) * radius
    ax.plot_wireframe(x, y, z, color="orange", alpha=0.1)

    # draw sphere at origin
    radius = 0.5
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v) * radius
    y = np.sin(u) * np.sin(v) * radius
    z = np.cos(v) * radius
    ax.plot_wireframe(x, y, z, color="orange", alpha=0.1)


def _draw_frustum(
    ax,
    camera: Camera,
    up="z",
    scene_radius=1.0
):
    if camera is None:
        return

    # get image plane corner points in 3D
    # from screen coordinates
    image_plane_vertices_2d = np.array(
        [[0, 0], [camera.height, 0], [0, camera.width], [camera.height, camera.width]]
    )

    rays_o, rays_d = get_rays_per_points_2d_screen(
        torch.from_numpy(camera.get_pose()).float(),
        torch.from_numpy(camera.get_intrinsics_inv()).float(),
        torch.from_numpy(image_plane_vertices_2d).float(),
    )
    # normalise to unit length
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()

    _draw_rays(
        ax,
        rays_o,
        rays_d,
        rgbs=np.zeros((rays_o.shape[0], 3)),
        masks=np.ones((rays_o.shape[0], 1)),
        up=up,
        scene_radius=scene_radius,
    )


def _draw_camera_frame(
    ax,
    pose: np.ndarray,
    idx=0,
    up="z",
    scene_radius=1.0
):
    if pose is None:
        return

    scale = scene_radius * SCALE_MULTIPLIER

    # get axis directions (normalized)
    x_dir = pose[:3, 0]
    x_dir /= np.linalg.norm(x_dir)
    y_dir = pose[:3, 1]
    y_dir /= np.linalg.norm(y_dir)
    z_dir = pose[:3, 2]
    z_dir /= np.linalg.norm(z_dir)
    # frame center
    pos = pose[:3, 3]

    # draw camera frame
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        x_dir[0],
        x_dir[1] if up == "z" else x_dir[2],
        x_dir[2] if up == "z" else x_dir[1],
        length=scale,
        color="r",
    )
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        y_dir[0],
        y_dir[1] if up == "z" else y_dir[2],
        y_dir[2] if up == "z" else y_dir[1],
        length=scale,
        color="g",
    )
    ax.quiver(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        z_dir[0],
        z_dir[1] if up == "z" else z_dir[2],
        z_dir[2] if up == "z" else z_dir[1],
        length=scale,
        color="b",
    )
    ax.text(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        str(idx),
    )


def _draw_point_clouds(
    ax,
    points_3d: list[np.ndarray] = [],
    points_3d_colors: list[np.ndarray] = [],
    points_3d_labels: list[str] = [],
    max_nr_points=1000,
    up="z",
    scene_radius=1.0
):
    if points_3d is None:
        return
    
    if not isinstance(points_3d, list):
        print_error("points_3d must be a list of numpy arrays")
        
    if len(points_3d) > 0:
        max_nr_points_per_pc = max_nr_points // len(points_3d)
        for pc, c, l in zip(points_3d, points_3d_colors, points_3d_labels):
            if max_nr_points_per_pc > 0:
                _draw_point_cloud(
                    ax,
                    points_3d=pc,
                    color=c,
                    label=l,
                    max_nr_points=max_nr_points_per_pc,
                    up=up,
                    scene_radius=scene_radius,
                )


def _draw_cameras(
    ax,
    cameras: list[Camera] = [],
    nr_rays=0,
    up="z",
    scene_radius=1.0,
    draw_image_planes=True,
    draw_cameras_frustums=True,
):
    if cameras is None:
        return
    
    if not isinstance(cameras, list):
        print_error("cameras must be a list of Cameras")

    if len(cameras) > 0:
        nr_rays_per_camera = nr_rays // len(cameras)

        # draw camera frames
        for camera in cameras:
            pose = camera.get_pose()
            camera_idx = camera.camera_idx
            _draw_camera_frame(
                ax, pose, idx=camera_idx, up=up, scene_radius=scene_radius
            )
            if draw_image_planes:
                _draw_image_plane(ax, camera, up=up, scene_radius=scene_radius)
            if draw_cameras_frustums:
                _draw_frustum(ax, camera, up=up, scene_radius=scene_radius)
            if nr_rays_per_camera > 0:
                _draw_camera_rays(
                    ax,
                    camera,
                    nr_rays=nr_rays_per_camera,
                    up=up,
                    scene_radius=scene_radius,
                )


def plot_3d(
    cameras: list[Camera] = [],
    points_3d: Union[list[np.ndarray], np.ndarray] = [],
    points_3d_colors: Union[list[np.ndarray], np.ndarray] = [],
    points_3d_labels: list[str] = [],
    bounding_boxes: list[BoundingBox] = [],
    bounding_spheres: list[BoundingSphere] = [],
    nr_rays=0,
    azimuth_deg=60,
    elevation_deg=30,
    scene_radius=1.0,
    up="z",
    draw_origin=True,
    draw_bounding_cube=True,
    draw_image_planes=True,
    draw_cameras_frustums=False,
    draw_contraction_spheres=False,
    figsize=(15, 15),
    title=None,
    show=True,
    save_path=None,  # if set, saves the figure to the given path
) -> None:
    """
    out:
        None
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    # get all camera poses
    poses = []
    for camera in cameras:
        poses.append(camera.get_pose())

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _draw_3d_init(
        ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
    )

    if draw_origin:
        _draw_cartesian_axis(ax, up=up, scene_radius=scene_radius)

    # draw points
    _draw_point_clouds(
        ax,
        points_3d=points_3d,
        points_3d_colors=points_3d_colors,
        points_3d_labels=points_3d_labels,
        max_nr_points=1000,
        up=up,
        scene_radius=scene_radius
    )

    # draw bounding cube
    if draw_bounding_cube:
        _draw_bounding_cube(ax)

    # draw camera frames
    _draw_cameras(
        ax,
        cameras,
        nr_rays=nr_rays,
        up=up,
        scene_radius=scene_radius,
        draw_image_planes=draw_image_planes,
        draw_cameras_frustums=draw_cameras_frustums,
    )

    # plot bounding boxes (if given)
    _draw_bounding_boxes(
        ax=ax,
        bounding_boxes=bounding_boxes,
        up=up,
        scene_radius=scene_radius,
        draw_frame=True
    )

    # plot bounding sphere (if given)
    _draw_bounding_spheres(
        ax=ax,
        bounding_spheres=bounding_spheres,
        up=up,
        scene_radius=scene_radius
    )

    if draw_contraction_spheres:
        _draw_contraction_spheres(ax)

    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )

    if show:
        plt.show()

    plt.close()


def _draw_camera_rays(
    ax,
    camera,
    nr_rays,
    frame_idx=0,
    up="z",
    scene_radius=1.0,
):
    rays_o, rays_d, points_2d_screen = camera.get_rays()
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()

    if not camera.has_rgbs():
        # color rays with their uv coordinates
        xy = points_2d_screen  # [:, [1, 0]]
        z = np.zeros((xy.shape[0], 1))
        rgbs = np.concatenate([xy, z], axis=1)
        rgbs[:, 0] /= np.max(rgbs[:, 0])
        rgbs[:, 1] /= np.max(rgbs[:, 1])
    else:
        # use frame rgb
        vals = camera.get_data(points_2d_screen, frame_idx=frame_idx)
        rgbs = vals["rgbs"].cpu().numpy()

    if not camera.has_masks():
        # set to ones
        masks = np.ones((camera.height, camera.width, 1)).reshape(-1, 1) * 0.5
    else:
        vals = camera.get_data(points_2d_screen)
        masks = vals["masks"].cpu().numpy()

    # draw rays
    _draw_rays(
        ax,
        rays_o,
        rays_d,
        rgbs=rgbs,
        masks=masks,
        max_nr_rays=nr_rays,
        up=up,
        scene_radius=scene_radius,
    )


def plot_current_batch(
    cameras,
    cameras_idx,
    rays_o,
    rays_d,
    rgbs=None,
    masks=None,
    bounding_boxes=[],
    azimuth_deg=60,
    elevation_deg=30,
    scene_radius=1.0,
    up="z",
    draw_origin=True,
    draw_bounding_cube=True,
    figsize=(15, 15),
    title=None,
    show=True,
    save_path=None,  # if set, saves the figure to the given path
) -> None:
    """
    out:
        None
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    # convert to numpy
    cameras_idx = cameras_idx.cpu().numpy()
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    if rgbs is not None:
        rgbs = rgbs.cpu().numpy()
    else:
        # if rgb is not given, color rays blue
        rgbs = np.zeros((rays_o.shape[0], 3))
        rgbs[:, 2] = 1.0
    if masks is not None:
        masks = masks.cpu().numpy()
    else:
        # if mask is not given, set to 0.5
        masks = np.ones((rays_o.shape[0], 1)) * 0.5

    # get unique camera idxs
    unique_cameras_idx = np.unique(cameras_idx, axis=0)

    # get all unique camera poses
    poses = []
    for idx in unique_cameras_idx:
        poses.append(cameras[idx].get_pose())

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _draw_3d_init(
        ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
    )

    if draw_origin:
        _draw_cartesian_axis(ax, up=up, scene_radius=scene_radius)

    # draw bounding cube
    if draw_bounding_cube:
        _draw_bounding_cube(ax)

    # plot unique camera poses
    for pose, camera_idx in zip(poses, unique_cameras_idx):
        _draw_camera_frame(ax, pose, idx=camera_idx, up=up, scene_radius=scene_radius)

    # draw rays
    _draw_rays(
        ax,
        rays_o,
        rays_d,
        rgbs=rgbs,
        masks=masks,
        max_nr_rays=None,
        up=up,
        scene_radius=scene_radius,
    )

    # plot bounding boxes (if given)
    _draw_bounding_boxes(ax, bounding_boxes, up=up, scene_radius=scene_radius)

    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )

    if show:
        plt.show()

    plt.close()


def plot_points_2d_on_image(
    camera,
    points_2d_screen,
    points_norms=None,
    frame_idx=0,
    show_ticks=False,
    figsize=(15, 15),
    title=None,
    show=True,
    save_path=None,  # if set, saves the figure to the given path
) -> None:
    """
    args:
        camera (Camera): camera object
        points_2d_screen (np.ndarray, float): (N, 2), (H, W)
        frame_idx (int, optional): Defaults to 0.
    out:
        None
    """
    if not camera.has_rgbs():
        raise ValueError("camera has no rgb modality")

    data = camera.get_data(keys=["rgbs", "masks"], frame_idx=frame_idx)
    rgb = data["rgbs"]
    mask = None
    if camera.has_masks():
        mask = data["masks"]
        rgb = rgb * np.clip(mask + 0.2, 0, 1)
    
    # reshape to (W, H, -1)
    rgb = rgb.reshape(camera.width, camera.height, -1)
    # transpose to (H, W, -1)
    rgb = np.transpose(rgb, (1, 0, 2))
    
    # init figure
    fig = plt.figure(figsize=figsize)
    
    if title is not None:
        plt.title(title)

    plt.imshow(rgb, alpha=0.8, resample=True)

    # Calculate (height_of_image / width_of_image)
    im_ratio = rgb.shape[0]/rgb.shape[1]
    
    # filter out points outside image range
    points_mask = get_mask_points_out_image(points_2d_screen, camera.width, camera.height)
    points_2d_screen = points_2d_screen[points_mask]

    if points_norms is None:
        # color points with their uv coordinates
        color = np.column_stack(
            [points_2d_screen, np.zeros((points_2d_screen.shape[0], 1))]
        )
        color[:, 0] /= camera.width
        color[:, 1] /= camera.height
    else:
        points_norms = points_norms[points_mask]
        # apply cmap to points norms
        from matplotlib import cm
        norm = plt.Normalize(vmin=np.min(points_norms), vmax=np.max(points_norms))
        cmap = cm.get_cmap("jet")
        color = cmap(norm(points_norms))
        # TODO: fix
        # make the colorbar for points_norms
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # Add a colorbar to the plot
        # plt.colorbar(sm, fraction=COLORBAR_FRACTION*im_ratio)
    points_2d_screen -= 0.5  # to avoid imshow shift
    
    plt.scatter(points_2d_screen[:, 0], points_2d_screen[:, 1], s=5, c=color, marker=".")
        
    # plt.gca().set_aspect("equal", adjustable="box")

    if show_ticks:
        plt.xticks(np.arange(-0.5, camera.width, 1), minor=True)
        plt.yticks(np.arange(-0.5, camera.height, 1), minor=True)
        plt.xticks(
            np.arange(-0.5, camera.width, 20),
            labels=np.arange(0.0, camera.width + 1, 20),
        )
        plt.yticks(
            np.arange(-0.5, camera.height, 20),
            labels=np.arange(0.0, camera.height + 1, 20),
        )
        plt.grid(which="minor", alpha=0.2)
        plt.grid(which="major", alpha=0.2)

    plt.xlabel("W")
    plt.ylabel("H")

    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )

    if show:
        plt.show()

    plt.close()


def plot_rays_samples(
    rays_o,
    rays_d,
    t_near,
    t_far,
    camera=None,
    bounding_boxes=[],
    bounding_spheres=[],
    points_samples=[],
    points_samples_colors=[],
    points_samples_sizes=[],
    points_samples_labels=[],
    azimuth_deg=60,
    elevation_deg=30,
    scene_radius=1.0,
    up="z",
    draw_origin=True,
    draw_contraction_spheres=True,
    draw_near_far_points=True,
    figsize=(15, 15),
    title=None,
    show=True,
    save_path=None,  # if set, saves the figure to the given path
) -> None:
    """
    out:
        None
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    assert (
        rays_o.shape[0] == rays_d.shape[0]
    ), "ray_o and ray_d must have the same length"
    assert (
        t_near.shape[0] == t_far.shape[0]
    ), "t_near and t_far must have the same length"
    assert (
        rays_o.shape[0] == t_near.shape[0]
    ), "ray_o and t_near must have the same length"

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _draw_3d_init(
        ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
    )

    if draw_origin:
        _draw_cartesian_axis(ax, up=up, scene_radius=scene_radius)

    # draw camera
    if camera is not None:
        _draw_cameras(
            ax,
            [camera],
            up=up,
            scene_radius=scene_radius,
            draw_image_planes=True,
            draw_cameras_frustums=True,
        )

    # plot bounding box
    for bb in bounding_boxes:
        _draw_bounding_box(ax, bb, up=up, scene_radius=scene_radius, draw_frame=False)

    # plot bounding spheres
    for sphere in bounding_spheres:
        _draw_bounding_sphere(
            ax, sphere, up=up, scene_radius=scene_radius, draw_frame=False
        )

    if draw_near_far_points:
        # draw t_near, t_far points
        p_near = rays_o + rays_d * t_near
        p_far = rays_o + rays_d * t_far

        # edge case of a single ray
        if p_near.ndim == 1:
            p_near = p_near.unsqueeze(1)
            p_far = p_far.unsqueeze(1)

        rays_o = rays_o.cpu().numpy()
        rays_d = rays_d.cpu().numpy()
        p_near = p_near.cpu().numpy()
        p_far = p_far.cpu().numpy()

        p_boundaries = np.concatenate(
            [p_near[:, np.newaxis, :], p_far[:, np.newaxis, :]], axis=1
        )

        for i in range(p_boundaries.shape[0]):
            # draw t_near, t_far points
            _draw_point_cloud(
                ax,
                p_boundaries.reshape(-1, 3),
                size=200,
                color="black",
                marker="x",
                up=up,
                scene_radius=scene_radius,
            )

    # draw points
    for i, samples in enumerate(points_samples):

        size = None
        if len(points_samples_sizes) > 0:
            assert len(points_samples) == len(
                points_samples_sizes
            ), "points_samples and points_samples_sizes must have the same length"
            size = points_samples_sizes[i]

        color = None
        if len(points_samples_colors) > 0:
            assert len(points_samples) == len(
                points_samples_colors
            ), "points_samples and points_samples_colors must have the same length"
            color = points_samples_colors[i]

        label = None
        if len(points_samples_labels) > 0:
            assert len(points_samples) == len(
                points_samples_labels
            ), "points_samples and points_samples_labels must have the same length"
            label = points_samples_labels[i]

        # draw sampling points
        _draw_point_cloud(
            ax,
            samples.cpu().numpy(),
            size=size,
            color=color,
            up=up,
            scene_radius=scene_radius,
            label=label,
        )

    if draw_contraction_spheres:
        _draw_contraction_spheres(ax)

    # Get current axes and check if there are any labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Only display legend if there are labels
    if labels:
        plt.legend()

    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )

    if show:
        plt.show()

    plt.close()


def plot_image(
    image: np.ndarray,  # (W, H)
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    draw_colorbar: bool = False,
    cmap: str = "viridis",
    figsize=(15, 15),
    show: bool = True,
    save_path: str = None,
):
    """Plots an image.

    Args:
        image (np.ndarray): (W, H) or (W, H, 1) or (W, H, 3) or (W, H, 4):.
        title (str, optional): Defaults to None.
    """
    
    # init figure
    fig = plt.figure(figsize=figsize)
    
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    # transpose to (H, W, C)
    image = np.transpose(image, (1, 0, 2))
    
    plt.imshow(image, cmap=cmap)
    
    # Calculate (height_of_image / width_of_image)
    im_ratio = image.shape[0]/image.shape[1]
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel("W")
        
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("H")
    
    if title is not None:
        plt.title(title)
    
    if draw_colorbar:
        plt.colorbar(fraction=COLORBAR_FRACTION*im_ratio)
    
    if save_path is not None:
        plt.savefig(
            save_path,
            transparent=TRANSPARENT,
            bbox_inches=BBOX_INCHES,
            pad_inches=PAD_INCHES,
            dpi=DPI,
        )

    if show:
        plt.show()
        
    plt.close()