import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations
from mvdatasets.utils.raycasting import (
    get_camera_rays,
    get_camera_frames_per_points_2d,
    get_camera_rays_per_points_2d,
)
from mvdatasets.utils.geometry import linear_transformation_3d, inv_perspective_projection

# from mvdatasets.scenes.camera import Camera
# import math

# Use in a notebook with:

# import ipywidgets as widgets
# from IPython.display import display
# @widgets.interact(azimuth_deg=(0, 360))
# def f(azimuth_deg=5):
#     plot_fn(..., azimuth_deg=azimuth_deg, ...)


def plot_cameras(
    cameras, points=None, azimuth_deg=60, elevation_deg=30, up="z", figsize=(15, 15), title=None
):
    """
    out:
        matplotlib figure
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    # Get all camera poses
    poses = []
    for camera in cameras:
        poses.append(camera.get_pose())
    poses = np.stack(poses, 0)

    # Get all camera centers
    camera_centers = poses[:, :3, 3]
    camera_distances_from_origin = np.linalg.norm(camera_centers, axis=1)
    scene_radius = max(np.max(camera_distances_from_origin) * 0.75, 1.0)
    scale = scene_radius * 0.2

    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_subplot(111, projection="3d")
    if title is not None:
        ax.set_title(title)

    # Cartesian axes
    ax.quiver(0, 0, 0, 1, 0, 0, length=scale, color="r")
    if up == "z":
        ax.quiver(0, 0, 0, 0, 1, 0, length=scale, color="g")  # matplotlib y
        ax.quiver(0, 0, 0, 0, 0, 1, length=scale, color="b")  # matplotlib z
    else:  # up == "y"
        ax.quiver(0, 0, 0, 0, 0, 1, length=scale, color="g")  # matplotlib y
        ax.quiver(0, 0, 0, 0, 1, 0, length=scale, color="b")  # matplotlib z
    ax.text(0, 0, 0, "w")

    # draw points
    if points is not None:
        if up == "z":
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)
        else:  # up = "y"
            ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=0.1)

    # Draw bounding cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="black")

    for i, pose in enumerate(poses):
        if up == "z":
            ax.quiver(
                pose[0, 3],
                pose[1, 3],
                pose[2, 3],
                pose[0, 0],
                pose[1, 0],
                pose[2, 0],
                length=scale,
                color="r",
            )
            ax.quiver(
                pose[0, 3],
                pose[1, 3],
                pose[2, 3],
                pose[0, 1],
                pose[1, 1],
                pose[2, 1],
                length=scale,
                color="g",
            )
            ax.quiver(
                pose[0, 3],
                pose[1, 3],
                pose[2, 3],
                pose[0, 2],
                pose[1, 2],
                pose[2, 2],
                length=scale,
                color="b",
            )
            ax.text(pose[0, 3], pose[1, 3], pose[2, 3], str(cameras[i].camera_idx))
        else:  
            # up = "y"
            ax.quiver(
                pose[0, 3],
                pose[2, 3],
                pose[1, 3],
                pose[0, 0],
                pose[2, 0],
                pose[1, 0],
                length=scale,
                color="r",
            )
            ax.quiver(
                pose[0, 3],
                pose[2, 3],
                pose[1, 3],
                pose[0, 1],
                pose[2, 1],
                pose[1, 1],
                length=scale,
                color="g",
            )
            ax.quiver(
                pose[0, 3],
                pose[2, 3],
                pose[1, 3],
                pose[0, 2],
                pose[2, 2],
                pose[1, 2],
                length=scale,
                color="b",
            )
            ax.text(pose[0, 3], pose[2, 3], pose[1, 3], str(cameras[i].camera_idx))

    lim = scene_radius
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-1, lim])

    ax.set_xlabel("X")
    if up == "z":
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

    # axis equal
    ax.set_aspect("equal")
    ax.view_init(elevation_deg, azimuth_deg)

    return fig


def plot_camera_rays(
    camera, nr_rays, azimuth_deg=60, elevation_deg=30, up="z", figsize=(15, 15)
):
    """
    out:
        matplotlib figure
    """

    if not (up == "z" or up == "y"):
        raise ValueError("up must be either 'y' or 'z'")

    # Get all camera poses
    pose = camera.get_pose()
    camera_center = pose[:3, 3]

    rays_o, rays_d, points_2d = get_camera_rays(camera, device="cpu")
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    
    # get frames
    rgb = camera.get_rgb()
    mask = camera.get_mask()
    vals = get_camera_frames_per_points_2d(points_2d, rgb=rgb, mask=mask)
    
    if "rgb" not in vals:
        # color rays with their uv coordinates
        xy = points_2d[:, [1, 0]]
        z = np.zeros((xy.shape[0], 1))
        rgb = np.concatenate([xy, z], axis=1)
        rgb[:, 0] /= np.max(rgb[:, 0])
        rgb[:, 1] /= np.max(rgb[:, 1])
    else:
        rgb = vals["rgb"]
        
    # # visualize rgb
    # plt.imshow(rgb.reshape(camera.height, camera.width, 3))
    # plt.show()
        
    if "mask" not in vals:
        # set to ones
        mask = np.ones((camera.height, camera.width, 1))
    else:
        mask = vals["mask"]
    
    # # visualize mask
    # plt.imshow(mask.reshape(camera.height, camera.width, 1))
    # plt.show()

    # subsample
    idx = np.random.permutation(rays_o.shape[0])[:nr_rays]
    rays_o = rays_o[idx]
    rays_d = rays_d[idx]
    rgb = rgb[idx]
    mask = mask[idx]
    
    # scene scale
    camera_distance_from_origin = np.linalg.norm(camera_center)
    scene_radius = max(camera_distance_from_origin * 0.75, 1.0)
    scale = scene_radius * 0.2

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Cartesian axes
    ax.quiver(0, 0, 0, 1, 0, 0, length=scale, color="r")
    if up == "z":
        ax.quiver(0, 0, 0, 0, 1, 0, length=scale, color="g")  # matplotlib y
        ax.quiver(0, 0, 0, 0, 0, 1, length=scale, color="b")  # matplotlib z
    else:  # up == "y"
        ax.quiver(0, 0, 0, 0, 0, 1, length=scale, color="g")  # matplotlib y
        ax.quiver(0, 0, 0, 0, 1, 0, length=scale, color="b")  # matplotlib z
    ax.text(0, 0, 0, "w")

    # Draw bounding cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="black")
    
    # Draw image plane
    
    # image plane distance
    image_plane_z = scale

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
        torch.from_numpy(pose).float(),
        torch.from_numpy(camera.get_intrinsics_inv()).float(),
        torch.from_numpy(corner_points_2d_screen).float()
    )
    corner_points_d = corner_points_d.cpu().numpy()

    corner_points_3d_world = camera_center + corner_points_d * image_plane_z
    
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
            
    # Draw camera frame
    if up == "z":
        ax.quiver(
            pose[0, 3],
            pose[1, 3],
            pose[2, 3],
            pose[0, 0],
            pose[1, 0],
            pose[2, 0],
            length=scale,
            color="r",
        )
        ax.quiver(
            pose[0, 3],
            pose[1, 3],
            pose[2, 3],
            pose[0, 1],
            pose[1, 1],
            pose[2, 1],
            length=scale,
            color="g",
        )
        ax.quiver(
            pose[0, 3],
            pose[1, 3],
            pose[2, 3],
            pose[0, 2],
            pose[1, 2],
            pose[2, 2],
            length=scale,
            color="b",
        )
        ax.text(pose[0, 3], pose[1, 3], pose[2, 3], str(camera.camera_idx))
    else:  # up = "y"
        ax.quiver(
            pose[0, 3],  # x
            pose[2, 3],  # z
            pose[1, 3],  # y
            pose[0, 0],  # dx
            pose[2, 0],  # dz
            pose[1, 0],  # dy
            length=scale,
            color="r",
        )
        ax.quiver(
            pose[0, 3],
            pose[2, 3],
            pose[1, 3],
            pose[0, 1],
            pose[2, 1],
            pose[1, 1],
            length=scale,
            color="g",
        )
        ax.quiver(
            pose[0, 3],
            pose[2, 3],
            pose[1, 3],
            pose[0, 2],
            pose[2, 2],
            pose[1, 2],
            length=scale,
            color="b",
        )
        ax.text(pose[0, 3], pose[2, 3], pose[1, 3], str(camera.camera_idx))

    # draw rays
    ray_lenght = scene_radius * 2
    for ray_o, ray_d, color, alpha in zip(rays_o, rays_d, rgb, mask):
        start_point = ray_o
        end_point = ray_o + ray_d * ray_lenght

        # plot line segment
        if up == "z":
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color,
                alpha=min(0.1, 0.25 * float(alpha)),
            )
        else:
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[2], end_point[2]],
                [start_point[1], end_point[1]],
                color=color,
                alpha=min(0.1, 0.25 * float(alpha)),
            )

    lim = scene_radius
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-1, lim])

    ax.set_xlabel("X")
    if up == "z":
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

    # axis equal
    ax.set_aspect("equal")
    ax.view_init(elevation_deg, azimuth_deg)

    return fig


def plot_current_batch(
    cameras,
    cameras_idx,
    rays_o,
    rays_d,
    rgb,
    mask,
    azimuth_deg=60,
    elevation_deg=30,
    up="z",
    figsize=(15, 15),
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
    rgb = rgb.cpu().numpy()
    mask = mask.cpu().numpy()

    # get camera poses in this batch
    poses = []
    for camera in cameras:
        poses.append(camera.get_pose())
    poses = np.stack(poses, 0)

    # get all camera centers
    camera_centers = poses[:, :3, 3]
    
    # this has to be computer over all cameras, not just those selected in the batch
    max_camera_distance_from_origin = np.max(np.linalg.norm(camera_centers, axis=1))
    scene_radius = max(max_camera_distance_from_origin * 0.75, 1.0)
    scale = scene_radius * 0.2

    # get unique camera idxs
    unique_cameras_idx = np.unique(cameras_idx, axis=0)
    
    # get all camera poses
    poses = []
    for idx in unique_cameras_idx:
        poses.append(cameras[idx].get_pose())
    poses = np.stack(poses, 0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Cartesian axes
    ax.quiver(0, 0, 0, 1, 0, 0, length=scale, color="r")
    if up == "z":
        ax.quiver(0, 0, 0, 0, 1, 0, length=scale, color="g")  # matplotlib y
        ax.quiver(0, 0, 0, 0, 0, 1, length=scale, color="b")  # matplotlib z
    else:  # up == "y"
        ax.quiver(0, 0, 0, 0, 0, 1, length=scale, color="g")  # matplotlib y
        ax.quiver(0, 0, 0, 0, 1, 0, length=scale, color="b")  # matplotlib z
    ax.text(0, 0, 0, "w")

    # Draw bounding cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="black", alpha=0.5)

    # get unique poses (not to draw multiple times the same camera frame)
    # poses = np.unique(poses, axis=0)
    for i, pose in enumerate(poses):
        idx = unique_cameras_idx[i]
        if up == "z":
            ax.quiver(
                pose[0, 3],
                pose[1, 3],
                pose[2, 3],
                pose[0, 0],
                pose[1, 0],
                pose[2, 0],
                length=scale,
                color="r",
            )
            ax.quiver(
                pose[0, 3],
                pose[1, 3],
                pose[2, 3],
                pose[0, 1],
                pose[1, 1],
                pose[2, 1],
                length=scale,
                color="g",
            )
            ax.quiver(
                pose[0, 3],
                pose[1, 3],
                pose[2, 3],
                pose[0, 2],
                pose[1, 2],
                pose[2, 2],
                length=scale,
                color="b",
            )
            ax.text(pose[0, 3], pose[1, 3], pose[2, 3], str(idx))
        else:  # up = "y"
            ax.quiver(
                pose[0, 3],
                pose[2, 3],
                pose[1, 3],
                pose[0, 0],
                pose[2, 0],
                pose[1, 0],
                length=scale,
                color="r",
            )
            ax.quiver(
                pose[0, 3],
                pose[2, 3],
                pose[1, 3],
                pose[0, 1],
                pose[2, 1],
                pose[1, 1],
                length=scale,
                color="g",
            )
            ax.quiver(
                pose[0, 3],
                pose[2, 3],
                pose[1, 3],
                pose[0, 2],
                pose[2, 2],
                pose[1, 2],
                length=scale,
                color="b",
            )
            ax.text(pose[0, 3], pose[2, 3], pose[1, 3], str(idx))

    # Draw rays
    ray_lenght = scene_radius * 2
    for ray_o, ray_d, color, alpha in zip(rays_o, rays_d, rgb, mask):
        start_point = ray_o
        end_point = ray_o + ray_d * ray_lenght

        # plot line segment
        if up == "z":
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color,
                alpha=0.3 * min(0.1, float(alpha)),
            )
        else:
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[2], end_point[2]],
                [start_point[1], end_point[1]],
                color=color,
                alpha=0.3 * max(0.1, float(alpha)),
            )

    lim = scene_radius
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-1, lim])

    ax.set_xlabel("X")
    if up == "z":
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

    # axis equal
    ax.set_aspect("equal")
    ax.view_init(elevation_deg, azimuth_deg)

    return fig


def plot_points_2d_on_image(
    camera, points_2d, frame_idx=0, show_ticks=False, figsize=(15, 15)
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
    
    fig = plt.figure(figsize=figsize)
    plt.imshow(rgb, alpha=0.8, resample=True)
    
    # filter out points outside image range
    points_2d = points_2d[points_2d[:, 0] >= 0]
    points_2d = points_2d[points_2d[:, 1] >= 0]
    points_2d = points_2d[points_2d[:, 0] < camera.width]
    points_2d = points_2d[points_2d[:, 1] < camera.height]

    rgb = np.column_stack([points_2d, np.zeros((points_2d.shape[0], 1))])
    rgb[:, 0] /= camera.width
    rgb[:, 1] /= camera.height
    points_2d -= 0.5  # to avoid imshow shift
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10, c=rgb, marker=".")
    plt.gca().set_aspect("equal", adjustable="box")
    
    print("camera.width", camera.width)
    print("camera.height", camera.height)
    
    if show_ticks:
        plt.xticks(np.arange(-0.5, camera.width, 1), minor=True)
        plt.yticks(np.arange(-0.5, camera.height, 1), minor=True)
        plt.xticks(np.arange(-0.5, camera.width, 20), labels=np.arange(0.0, camera.width+1, 20))
        plt.yticks(np.arange(-0.5, camera.height, 20), labels=np.arange(0.0, camera.height+1, 20))
        plt.grid(which="minor", alpha=0.2)
        plt.grid(which="major", alpha=0.2)
    
    plt.xlabel("x")
    plt.ylabel("y")

    return fig
