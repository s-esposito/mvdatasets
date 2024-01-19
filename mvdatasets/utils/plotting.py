import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
from mvdatasets.utils.raycasting import (
    get_camera_rays,
    get_camera_frames_per_points_2d,
    get_camera_rays_per_points_2d,
)

# from mvdatasets.scenes.camera import Camera
# import math

# Use in a notebook with:

# import ipywidgets as widgets
# from IPython.display import display
# @widgets.interact(azimuth_deg=(0, 360))
# def f(azimuth_deg=5):
#     plot_fn(..., azimuth_deg=azimuth_deg, ...)


def _scene_radius(poses):
    """
    compute scene radius from list of poses

    Args:
        poses (list): list of numpy (4, 4) poses

    Returns:
        scene_radius (float): scene radius
    """
    # get all camera centers
    camera_centers = np.stack(poses, 0)[:, :3, 3]
    camera_distances_from_origin = np.linalg.norm(camera_centers, axis=1)
    scene_radius = max(np.max(camera_distances_from_origin) * 0.75, 1.0)
    return scene_radius


def _plot_3d_init(ax, scene_radius=1, elevation_deg=60, azimuth_deg=30, up="z"):
    lim = scene_radius
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-1, lim])

    ax.set_xlabel("X")
    ax.set_ylabel("Y") if up == "z" else ax.set_ylabel("Z")
    ax.set_zlabel("Z") if up == "z" else ax.set_zlabel("Y")

    # axis equal
    ax.set_aspect("equal")
    ax.view_init(elevation_deg, azimuth_deg)


def _plot_bounding_cube(ax, side_lenght=1, up="z", scale=1.0):
    # draw bounding cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="black")


def _plot_rays(ax, rays_o, rays_d, rgb=None, mask=None, max_nr_rays=None, ray_lenght=1, up="z", scale=1.0):
    if rays_o is None or rays_d is None:
        return
    
    # subsample
    if max_nr_rays is not None:
        if max_nr_rays < rays_o.shape[0]:
            idx = np.random.permutation(rays_o.shape[0])[:max_nr_rays]
            rays_o = rays_o[idx]
            rays_d = rays_d[idx]
            if rgb is not None:
                rgb = rgb[idx]
            if mask is not None:
                mask = mask[idx]
    
    # draw rays
    for i, (ray_o, ray_d) in enumerate(zip(rays_o, rays_d)):
        start_point = ray_o
        end_point = ray_o + ray_d * ray_lenght
        color = rgb[i] if rgb is not None else "blue"
        alpha = mask[i] if mask is not None else 0.5
        # plot line segment
        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]] if up == "z" else [start_point[2], end_point[2]],
            [start_point[2], end_point[2]] if up == "z" else [start_point[1], end_point[1]],
            color=color,
            alpha=0.3 * float(alpha),
        )


def _plot_point_cloud(ax, points_3d, max_nr_points=None, up="z", scale=1.0):
    if points_3d is None:
        return
    
    # subsample
    if max_nr_points is not None:
        if max_nr_points < points_3d.shape[0]:
            idx = np.random.permutation(points_3d.shape[0])[:max_nr_points]
            points_3d = points_3d[idx]
        
    # draw points
    if up == "z":
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=0.1)
    else:  # up = "y"
        ax.scatter(points_3d[:, 0], points_3d[:, 2], points_3d[:, 1], s=0.1)


def _plot_frame(ax, pose, idx=0, up="z", scale=1.0):
    if pose is None:
        return

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
    ax.text(
        pos[0],  # x
        pos[1] if up == "z" else pos[2],  # y
        pos[2] if up == "z" else pos[1],  # z
        str(idx)
    )


def _plot_cartesian_axis(ax, up="z", scale=1.0):
    _plot_frame(ax, np.eye(4), idx="w", up=up, scale=scale)


def _plot_bounding_box(ax, bb, idx=0, up="z", scale=1.0):
    if bb is None:
        return
    
    # draw bounding box
    segments_indices = np.array([
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
        [6, 7]
    ])
    
    # get bb pose
    pose = bb.get_pose()
    
    # get vertices and pairs of vertices for plotting
    vertices = bb.get_vertices()
    vertices_pairs = vertices[segments_indices]
    
    # plot line segments
    for pair in vertices_pairs:
        ax.plot3D(
            *zip(
                pair[0] if up == "z" else pair[0][[0, 2, 1]],
                pair[1] if up == "z" else pair[1][[0, 2, 1]]
            ),
            color="black",
            linewidth=1.0,
            alpha=0.5
        )
    
    # draw bb frame
    _plot_frame(ax, pose, idx=idx, up=up, scale=scale)


def _plot_bounding_boxes(ax, bounding_boxes, up="z", scale=1.0):
    if bounding_boxes is None:
        return
    
    # draw bounding boxes
    for i, bb in enumerate(bounding_boxes):
        _plot_bounding_box(ax, bb, idx=i, up=up, scale=scale)


def _plot_image_plane(ax, camera, up="z", scale=1.0):
    if camera is None:
        return
    
    # image plane distance
    image_plane_z = 1.1 * scale

    # get image plane corner points in 3D
    # from screen coordinates
    eps = 1e-6
    corner_points_2d_screen = np.array(
                                    [
                                        [0, 0],
                                        [camera.height-eps, 0],
                                        [0, camera.width-eps],
                                        [camera.height-eps, camera.width-eps]
                                    ]
                                )
    
    _, corner_points_d = get_camera_rays_per_points_2d(
        torch.from_numpy(camera.get_pose()).float(),
        torch.from_numpy(camera.get_intrinsics_inv()).float(),
        torch.from_numpy(corner_points_2d_screen).float()
    )
    corner_points_d = corner_points_d.cpu().numpy()

    corner_points_3d_world = camera.get_center() + corner_points_d * image_plane_z
    
    for i, j in combinations(range(4), 2):
        if up == "z":
            ax.plot3D(
                        *zip(
                            corner_points_3d_world[i],
                            corner_points_3d_world[j]),
                        color="black",
                        linewidth=1.0,
                        alpha=0.5
                    )
        else:
            ax.plot3D(
                        *zip(
                            corner_points_3d_world[:, [0, 2, 1]][i],
                            corner_points_3d_world[:, [0, 2, 1]][j]),
                        color="black",
                        linewidth=1.0,
                        alpha=0.5
                    )


def _plot_camera_frame(ax, pose, idx=0, up="z", scale=1.0):
    if pose is None:
        return
    
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
        str(idx)
    )


def _plot_camera_frames(ax, poses, camera_idxs, up="z", scale=1.0):
    if poses is None or camera_idxs is None:
        return
    # draw camera frames
    for pose, camera_idx in zip(poses, camera_idxs):
        _plot_camera_frame(ax, pose, idx=camera_idx, up=up, scale=scale)


def plot_cameras(
    cameras,
    points_3d=None,
    bounding_boxes=[],
    azimuth_deg=60,
    elevation_deg=30,
    up="z",
    figsize=(15, 15),
    title=None
):
    """
    out:
        matplotlib figure
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    # get all camera poses
    poses = []
    camera_idxs = []
    for camera in cameras:
        poses.append(camera.get_pose())
        camera_idxs.append(camera.camera_idx)

    # scene radius and scale
    scene_radius = _scene_radius(poses)
    scale = scene_radius * 0.1

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _plot_3d_init(
        ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg
    )

    _plot_cartesian_axis(ax, up=up, scale=scale)

    # draw points
    _plot_point_cloud(ax, points_3d, max_nr_points=1000, up=up, scale=scale)

    # draw bounding cube
    _plot_bounding_cube(ax, up=up, scale=scale)

    # draw camera frames
    _plot_camera_frames(ax, poses, camera_idxs, up=up, scale=scale)
    
    # plot bounding boxes (if given)
    _plot_bounding_boxes(ax, bounding_boxes, up=up, scale=scale)

    return fig


def plot_camera_rays(
    camera,
    nr_rays,
    points_3d=None,
    bounding_boxes=[],
    azimuth_deg=60,
    elevation_deg=30,
    up="z",
    figsize=(15, 15),
    title=None
):
    """
    out:
        matplotlib figure
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    # Get all camera poses
    pose = camera.get_pose()
    
    # scene radius and scale
    scene_radius = _scene_radius([pose])
    scale = scene_radius * 0.1

    rays_o, rays_d, points_2d = get_camera_rays(camera, device="cpu")
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    
    # get frames
    rgb = camera.get_rgb()
    mask = camera.get_mask()
    
    if rgb is None:
        # color rays with their uv coordinates
        xy = points_2d[:, [1, 0]]
        z = np.zeros((xy.shape[0], 1))
        rgb = np.concatenate([xy, z], axis=1)
        rgb[:, 0] /= np.max(rgb[:, 0])
        rgb[:, 1] /= np.max(rgb[:, 1])
    else:
        vals = get_camera_frames_per_points_2d(points_2d, rgb=rgb)
        rgb = vals["rgb"]
        
    if mask is None:
        # set to ones
        mask = np.ones((camera.height, camera.width, 1)).reshape(-1, 1) * 0.5
    else:
        vals = get_camera_frames_per_points_2d(points_2d, mask=mask)
        mask = vals["mask"]

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _plot_3d_init(
        ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg
    )

    # draw world origin
    _plot_cartesian_axis(ax, up=up, scale=scale)
    
    # draw points
    _plot_point_cloud(ax, points_3d, max_nr_points=1000, up=up, scale=scale)

    # draw bounding cube
    _plot_bounding_cube(ax, up=up, scale=scale)
    
    # draw camera frame
    _plot_camera_frame(ax, pose, camera.camera_idx, up=up, scale=scale)

    # draw rays
    _plot_rays(
        ax,
        rays_o,
        rays_d,
        rgb=rgb,
        mask=mask,
        max_nr_rays=nr_rays,
        ray_lenght=scene_radius * 2,
        up=up,
        scale=scale
    )

    # draw image plane
    _plot_image_plane(ax, camera, up=up, scale=scale)
    
    # plot bounding boxes (if given)
    _plot_bounding_boxes(ax, bounding_boxes, up=up, scale=scale)

    return fig


def plot_current_batch(
    cameras,
    cameras_idx,
    rays_o,
    rays_d,
    rgb=None,
    mask=None,
    azimuth_deg=60,
    elevation_deg=30,
    up="z",
    figsize=(15, 15),
    title=None
):
    """
    out:
        matplotlib figure
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    # convert to numpy
    cameras_idx = cameras_idx.cpu().numpy()
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    if rgb is not None:
        rgb = rgb.cpu().numpy()
    else:
        # if rgb is not given, color rays blue
        rgb = np.zeros((rays_o.shape[0], 3))
        rgb[:, 2] = 1.0
    if mask is not None:
        mask = mask.cpu().numpy()
    else:
        # if mask is not given, set to ones
        mask = np.ones((rays_o.shape[0], 1)) * 0.5

    # get unique camera idxs
    unique_cameras_idx = np.unique(cameras_idx, axis=0)
    
    # get all unique camera poses
    poses = []
    for idx in unique_cameras_idx:
        poses.append(cameras[idx].get_pose())
    
    # get all camera centers
    scene_radius = _scene_radius(poses)
    scale = scene_radius * 0.1

    # init figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)
    _plot_3d_init(
        ax,
        scene_radius=scene_radius,
        up=up,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg
    )

    _plot_cartesian_axis(ax, up=up, scale=scale)

    # draw bounding cube
    _plot_bounding_cube(ax, up=up, scale=scale)

    # plot unique camera poses
    _plot_camera_frames(ax, poses, unique_cameras_idx, up=up, scale=scale)

    # draw rays
    _plot_rays(
        ax,
        rays_o,
        rays_d,
        rgb=rgb,
        mask=mask,
        max_nr_rays=None,
        ray_lenght=scene_radius * 2,
        up=up,
        scale=scale
    )

    return fig


def plot_points_2d_on_image(
    camera,
    points_2d,
    points_norms=None,
    frame_idx=0,
    show_ticks=False,
    figsize=(15, 15),
    title=None
):
    """

    args:
        camera (Camera): camera object
        points_2d (np.ndarray, float): (N, 2) -> (x, y)
        frame_idx (int, optional): Defaults to 0.

    out:
        matplotlib figure
    """
    if not camera.has_rgbs():
        raise ValueError("camera has no rgb modality")
    
    rgb = camera.get_rgb(frame_idx=frame_idx)
    mask = None
    if camera.has_masks():
        mask = camera.get_mask(frame_idx=frame_idx)
        rgb = rgb * np.clip(mask + 0.2, 0, 1)
    print("rgb", rgb.shape)
    
    # init figure
    fig = plt.figure(figsize=figsize)
    if title is not None:
        fig.suptitle(title)
        
    plt.imshow(rgb, alpha=0.8, resample=True)
    
    # filter out points outside image range
    points_mask = points_2d[:, 0] >= 0
    points_mask *= points_2d[:, 1] >= 0
    points_mask *= points_2d[:, 0] < camera.width
    points_mask *= points_2d[:, 1] < camera.height
    points_2d = points_2d[points_mask]
    
    if points_norms is not None:
        points_norms = points_norms[points_mask]
        print("min dist", np.min(points_norms))
        print("max dist", np.max(points_norms))
    
    # points_2d = points_2d[points_2d[:, 0] >= 0]
    # points_2d = points_2d[points_2d[:, 1] >= 0]
    # points_2d = points_2d[points_2d[:, 0] < camera.width]
    # points_2d = points_2d[points_2d[:, 1] < camera.height]

    if points_norms is None:
        rgb = np.column_stack([points_2d, np.zeros((points_2d.shape[0], 1))])
        rgb[:, 0] /= camera.width
        rgb[:, 1] /= camera.height
    else:
        # apply cmap to points norms
        from matplotlib import cm
        norm = plt.Normalize(vmin=np.min(points_norms), vmax=np.max(points_norms))
        cmap = cm.get_cmap("jet")
        rgb = cmap(norm(points_norms))
    points_2d -= 0.5  # to avoid imshow shift
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10, c=rgb, marker=".")
    plt.gca().set_aspect("equal", adjustable="box")
    
    if show_ticks:
        plt.xticks(np.arange(-0.5, camera.width, 1), minor=True)
        plt.yticks(np.arange(-0.5, camera.height, 1), minor=True)
        plt.xticks(np.arange(-0.5, camera.width, 20), labels=np.arange(0.0, camera.width+1, 20))
        plt.yticks(np.arange(-0.5, camera.height, 20), labels=np.arange(0.0, camera.height+1, 20))
        plt.grid(which="minor", alpha=0.2)
        plt.grid(which="major", alpha=0.2)
    
    # if points_norms is not None:
    #     plt.colorbar()
    
    plt.xlabel("x")
    plt.ylabel("y")

    return fig
