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

def _scene_radius_to_scale(scene_radius):
    return scene_radius * 0.1

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
    scene_radius = np.max(camera_distances_from_origin)
    # scene_radius = max(np.max(camera_distances_from_origin) * 0.75, 1.0)
    return scene_radius


def _draw_3d_init(ax, scene_radius=1, elevation_deg=60, azimuth_deg=30, up="z"):
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


def _draw_bounding_cube(ax):
    # draw bounding cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="black")


def _draw_rays(ax, rays_o, rays_d, rgb=None, mask=None, max_nr_rays=None, ray_lenght=None, up="z", scene_radius=1.0):
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
    
    if ray_lenght is None:
        ray_lenght = 2 * scene_radius
    
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


def _draw_point_cloud(ax, points_3d, size=None, color=None, marker=None, max_nr_points=None, up="z", scene_radius=1.0):
    if points_3d is None:
        return
    
    # subsample
    if max_nr_points is not None:
        if max_nr_points < points_3d.shape[0]:
            idx = np.random.permutation(points_3d.shape[0])[:max_nr_points]
            points_3d = points_3d[idx]
    
    scale = _scene_radius_to_scale(scene_radius)
    
    if color is None:
        # random matplotlib color
        color = np.random.rand(3)
    if size is None:
        size = 10
    if marker is None:
        marker = "o"
    
    # draw points
    if up == "z":
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=scale*size, c=color, marker=marker)
    else:  # up = "y"
        ax.scatter(points_3d[:, 0], points_3d[:, 2], points_3d[:, 1], s=scale*size, c=color, marker=marker)


def _draw_frame(ax, pose, idx=0, up="z", scene_radius=1.0):
    if pose is None:
        return
    
    scale = _scene_radius_to_scale(scene_radius)

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
        str(idx)
    )


def _draw_cartesian_axis(ax, up="z", scene_radius=1.0):
    _draw_frame(ax, np.eye(4), idx="w", up=up, scene_radius=scene_radius)


def _draw_bounding_box(ax, bb, idx=0, up="z", scene_radius=1.0, draw_frame=False):
    if bb is None:
        return
    
    scale = _scene_radius_to_scale(scene_radius)
    
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
    ax.scatter(min_vertex[0], min_vertex[1], min_vertex[2], s=scale*50, color=color, marker="o")
    ax.scatter(max_vertex[0], max_vertex[1], max_vertex[2], s=scale*50, color=color, marker="o")
    
    # plot line segments
    for pair in vertices_pairs:
        ax.plot3D(
            *zip(
                pair[0] if up == "z" else pair[0][[0, 2, 1]],
                pair[1] if up == "z" else pair[1][[0, 2, 1]]
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


def _draw_bounding_boxes(ax, bounding_boxes, up="z", scene_radius=1.0, draw_frame=False):
    if bounding_boxes is None:
        return
    
    # draw bounding boxes
    for i, bb in enumerate(bounding_boxes):
        _draw_bounding_box(ax, bb, idx=i, up=up, scene_radius=scene_radius, draw_frame=draw_frame)


def _draw_image_plane(ax, camera, up="z", scene_radius=1.0):
    if camera is None:
        return

    scale = _scene_radius_to_scale(scene_radius)

    # get image plane corner points in 3D
    # from screen coordinates
    corner_points_2d_screen = np.array(
                                    [
                                        [0, 0],
                                        [camera.height, 0],
                                        [0, camera.width],
                                        [camera.height, camera.width]
                                    ]
                                )
    
    _, corner_points_d = get_camera_rays_per_points_2d(
        torch.from_numpy(camera.get_pose()).float(),
        torch.from_numpy(camera.get_intrinsics_inv()).float(),
        torch.from_numpy(corner_points_2d_screen).float()
    )
    corner_points_d = corner_points_d.cpu().numpy()

    corner_points_3d_world = camera.get_center() + corner_points_d * scale
    
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


def _draw_contraction_spheres(ax):
    
    # draw sphere at origin
    radius = 1.0
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)*radius
    y = np.sin(u)*np.sin(v)*radius
    z = np.cos(v)*radius
    ax.plot_wireframe(x, y, z, color="black", alpha=0.1)
    
    # draw sphere at origin
    radius = 0.5
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)*radius
    y = np.sin(u)*np.sin(v)*radius
    z = np.cos(v)*radius
    ax.plot_wireframe(x, y, z, color="black", alpha=0.1)
    

def _draw_frustum(ax, camera, up="z", scene_radius=1.0):
    if camera is None:
        return

    # get image plane corner points in 3D
    # from screen coordinates
    image_plane_vertices_2d = np.array(
                                    [
                                        [0, 0],
                                        [camera.height, 0],
                                        [0, camera.width],
                                        [camera.height, camera.width]
                                    ]
                                )
    
    rays_o, rays_d = get_camera_rays_per_points_2d(
        torch.from_numpy(camera.get_pose()).float(),
        torch.from_numpy(camera.get_intrinsics_inv()).float(),
        torch.from_numpy(image_plane_vertices_2d).float()
    )
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()

    _draw_rays(
        ax,
        rays_o,
        rays_d,
        rgb=np.zeros((rays_o.shape[0], 3)),
        up=up,
        scene_radius=scene_radius
    )


def _draw_camera_frame(ax, pose, idx=0, up="z", scene_radius=1.0):
    if pose is None:
        return
    
    scale = _scene_radius_to_scale(scene_radius)
    
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
    

def _draw_point_clouds(
    ax,
    point_clouds,
    max_nr_points=1000,
    up="z",
    scene_radius=1.0
):
    if not isinstance(point_clouds, list):
        point_clouds_ = [point_clouds]
    else:
        point_clouds_ = point_clouds
    
    if len(point_clouds_) > 0:
        max_nr_points_per_pc = max_nr_points // len(point_clouds_)
        for pc in point_clouds_:
            if max_nr_points_per_pc > 0:
                _draw_point_cloud(ax, pc, max_nr_points=max_nr_points_per_pc, up=up, scene_radius=scene_radius)


def _draw_cameras(
    ax,
    cameras,
    nr_rays=0,
    up="z",
    scene_radius=1.0,
    draw_image_planes=True,
    draw_cameras_frustums=True
):
    if not isinstance(cameras, list):
        cameras_ = [cameras]
    else:
        cameras_ = cameras
    
    if len(cameras_) > 0:
        nr_rays_per_camera = nr_rays // len(cameras_)
        
        # draw camera frames
        for camera in cameras_:
            pose = camera.get_pose()
            camera_idx = camera.camera_idx
            _draw_camera_frame(ax, pose, idx=camera_idx, up=up, scene_radius=scene_radius)
            if draw_image_planes:
                _draw_image_plane(ax, camera, up=up, scene_radius=scene_radius)
            if draw_cameras_frustums:
                _draw_frustum(ax, camera, up=up, scene_radius=scene_radius)
            if nr_rays_per_camera > 0:
                _draw_camera_rays(ax, camera, nr_rays=nr_rays_per_camera, up=up, scene_radius=scene_radius)


def plot_cameras(
    cameras,
    points_3d=None,
    bounding_boxes=[],
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
        azimuth_deg=azimuth_deg
    )

    if draw_origin:
        _draw_cartesian_axis(ax, up=up, scene_radius=scene_radius)

    # draw points
    _draw_point_cloud(ax, points_3d, max_nr_points=1000, up=up, scene_radius=scene_radius)

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
        draw_cameras_frustums=draw_cameras_frustums
    )
    
    # plot bounding boxes (if given)
    _draw_bounding_boxes(ax, bounding_boxes, up=up, scene_radius=scene_radius)

    if draw_contraction_spheres:
        _draw_contraction_spheres(ax)
    
    return fig


def plot_bounding_boxes(
    bounding_boxes=[],
    cameras=[],
    point_clouds=[],
    azimuth_deg=60,
    elevation_deg=30,
    scene_radius=1.0,
    up="z",
    max_nr_points=1000,
    draw_origin=True,
    draw_frame=False,
    figsize=(15, 15),
    title=None
):
    """
    out:
        matplotlib figure
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

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
        azimuth_deg=azimuth_deg
    )

    # draw world origin
    if draw_origin:
        _draw_cartesian_axis(ax, up=up, scene_radius=scene_radius)
    
    # plot bounding boxes (if given)
    _draw_bounding_boxes(
        ax,
        bounding_boxes,
        up=up,
        scene_radius=scene_radius,
        draw_frame=draw_frame
    )
    
    # draw points
    _draw_point_clouds(ax, point_clouds, max_nr_points=max_nr_points, up=up, scene_radius=scene_radius)
    
    # draw cameras
    _draw_cameras(ax, cameras, up=up, scene_radius=scene_radius)
    
    return fig


def _draw_camera_rays(
    ax,
    camera,
    nr_rays,
    frame_idx=0,
    up="z",
    scene_radius=1.0,
):
    rays_o, rays_d, points_2d = get_camera_rays(camera, device="cpu")
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    
    if not camera.has_rgbs():
        # color rays with their uv coordinates
        xy = points_2d[:, [1, 0]]
        z = np.zeros((xy.shape[0], 1))
        rgb = np.concatenate([xy, z], axis=1)
        rgb[:, 0] /= np.max(rgb[:, 0])
        rgb[:, 1] /= np.max(rgb[:, 1])
    else:
        # use frame rgb
        rgb = camera.get_rgb(frame_idx=frame_idx)
        vals = get_camera_frames_per_points_2d(points_2d, rgb=rgb)
        rgb = vals["rgb"]
        
    if not camera.has_masks():
        # set to ones
        mask = np.ones((camera.height, camera.width, 1)).reshape(-1, 1) * 0.5
    else:
        mask = camera.get_mask(frame_idx=frame_idx)
        vals = get_camera_frames_per_points_2d(points_2d, mask=mask)
        mask = vals["mask"]

    # draw rays
    _draw_rays(
        ax,
        rays_o,
        rays_d,
        rgb=rgb,
        mask=mask,
        max_nr_rays=nr_rays,
        up=up,
        scene_radius=scene_radius
    )


def plot_current_batch(
    cameras,
    cameras_idx,
    rays_o,
    rays_d,
    rgb=None,
    mask=None,
    bounding_boxes=[],
    azimuth_deg=60,
    elevation_deg=30,
    scene_radius=1.0,
    up="z",
    draw_origin=True,
    draw_bounding_cube=True,
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
        # if mask is not given, set to 0.5
        mask = np.ones((rays_o.shape[0], 1)) * 0.5

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
        azimuth_deg=azimuth_deg
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
        rgb=rgb,
        mask=mask,
        max_nr_rays=None,
        up=up,
        scene_radius=scene_radius
    )
    
    # plot bounding boxes (if given)
    _draw_bounding_boxes(ax, bounding_boxes, up=up, scene_radius=scene_radius)

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
    
    rgb = camera.get_rgb(frame_idx=frame_idx) / 255.0
    mask = None
    if camera.has_masks():
        mask = camera.get_mask(frame_idx=frame_idx) / 255.0
        rgb = rgb * np.clip(mask + 0.2, 0, 1)
    # print("rgb", rgb.shape)
    
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


def plot_ray_samples(
    ray_o,
    ray_d,
    t_near,
    t_far,
    bounding_box,
    points_samples=[],
    points_samples_colors=[],
    points_samples_sizes=[],
    azimuth_deg=60,
    elevation_deg=30,
    scene_radius=1.0,
    up="z",
    draw_origin=True,
    draw_contraction_spheres=True,
    figsize=(15, 15),
    title=None
):
    """
    out:
        matplotlib figure
    """
    
    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")
    
    assert ray_o.shape[0] == 1, "ray_o must have shape (1, 3)"
    assert ray_o.shape[0] == ray_d.shape[0], "ray_o and ray_d must have the same length"
    
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
        azimuth_deg=azimuth_deg
    )

    if draw_origin:
        _draw_cartesian_axis(ax, up=up, scene_radius=scene_radius)

    # plot bounding box
    _draw_bounding_box(
        ax,
        bounding_box,
        up=up,
        scene_radius=scene_radius,
        draw_frame=False
    )
    
    # draw t_near, t_far points
    p_near = ray_o + ray_d * t_near
    p_far = ray_o + ray_d * t_far
    p_near = p_near.unsqueeze(1)
    p_far = p_far.unsqueeze(1)
    
    ray_o = ray_o.cpu().numpy()
    ray_d = ray_d.cpu().numpy()
    p_near = p_near.cpu().numpy()
    p_far = p_far.cpu().numpy()
    
    p_boundaries = np.concatenate([p_near, p_far], axis=1)
    print(p_boundaries.shape)
    
    for i in range(p_boundaries.shape[0]):
        # draw t_near, t_far points
        _draw_point_cloud(
            ax,
            p_boundaries[0],
            size=200,
            color="black",
            marker="x",
            up=up,
            scene_radius=scene_radius
        )
    
    # draw points
    for i, samples in enumerate(points_samples):
        
        size = None
        if len(points_samples_sizes) > 0:
            assert len(points_samples) == len(points_samples_sizes), "points_samples and points_samples_sizes must have the same length"
            size = points_samples_sizes[i]
        
        color = None
        if len(points_samples_colors) > 0:
            assert len(points_samples) == len(points_samples_colors), "points_samples and points_samples_colors must have the same length"
            color = points_samples_colors[i]
        
        # draw sampling points
        _draw_point_cloud(
            ax,
            samples.cpu().numpy(),
            size=size,
            color=color,
            up=up,
            scene_radius=scene_radius
        )

    if draw_contraction_spheres:
        _draw_contraction_spheres(ax)

    return fig