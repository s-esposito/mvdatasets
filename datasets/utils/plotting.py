import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from itertools import product, combinations

from datasets.scenes.camera import Camera

# Use in a notebook with:

# import ipywidgets as widgets
# from IPython.display import display
# @widgets.interact(azimuth_deg=(0, 360))
# def f(azimuth_deg=5):
#     plot_fn(..., azimuth_deg=azimuth_deg, ...)


def plot_cameras(
    cameras, points=None, azimuth_deg=60, elevation_deg=30, up="z", figsize=(15, 15)
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
        poses.append(camera.get_pose().cpu().numpy())
    poses = np.stack(poses, 0)

    # Get all camera centers
    camera_centers = poses[:, :3, 3]

    scene_radius = np.max(np.linalg.norm(camera_centers, axis=1))
    scale = scene_radius * 0.1

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
            ax.text(pose[0, 3], pose[1, 3], pose[2, 3], str(i))
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
            ax.text(pose[0, 3], pose[2, 3], pose[1, 3], str(i))

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
    pose = camera.get_pose().cpu().numpy()

    pixels = camera.get_random_pixels(nr_rays)
    rays_o, rays_d = camera.get_rays_per_pixels(pixels)
    rgb, mask = camera.get_frame_per_pixels(pixels)

    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    rgb = rgb.cpu().numpy()
    mask = mask.cpu().numpy()

    # Get all camera centers
    camera_center = pose[:3, 3]

    scene_radius = np.linalg.norm(camera_center)
    scale = scene_radius * 0.1

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
        ax.text(pose[0, 3], pose[1, 3], pose[2, 3], "c")
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
        ax.text(pose[0, 3], pose[2, 3], pose[1, 3], "c")

    # Draw rays
    ray_lenght = scene_radius * 1.5
    for origin, dir, color, alpha in zip(rays_o, rays_d, rgb, mask):
        start_point = origin
        end_point = origin + dir * ray_lenght

        # plot line segment
        if up == "z":
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color,
                alpha=0.3 * min(0.25, float(alpha)),
            )
        else:
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[2], end_point[2]],
                [start_point[1], end_point[1]],
                color=color,
                alpha=0.3 * max(0.25, float(alpha)),
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

    # Convert to numpy
    cameras_idx = cameras_idx.cpu().numpy()
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()
    rgb = rgb.cpu().numpy()
    mask = mask.cpu().numpy()

    # Get unique camera idxs
    unique_cameras_idx = np.unique(cameras_idx, axis=0)

    # Get all camera poses
    poses = []
    for idx in unique_cameras_idx:
        poses.append(cameras[idx].get_pose().cpu().numpy())
    poses = np.stack(poses, 0)

    # Get all camera centers
    camera_centers = poses[:, :3, 3]

    scene_radius = np.max(np.linalg.norm(camera_centers, axis=1))
    scale = scene_radius * 0.1

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
    ray_lenght = scene_radius * 1.5
    for origin, dir, color, alpha in zip(rays_o, rays_d, rgb, mask):
        start_point = origin
        end_point = origin + dir * ray_lenght

        # plot line segment
        if up == "z":
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]],
                color=color,
                alpha=0.3 * min(0.25, float(alpha)),
            )
        else:
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[2], end_point[2]],
                [start_point[1], end_point[1]],
                color=color,
                alpha=0.3 * max(0.25, float(alpha)),
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


# TODO: plot_projected_points
# camera_idx = 3
# img_np = dataset_train.cameras[camera_idx].get_frame().cpu().numpy()
# intrinsics = dataset_train.cameras[camera_idx].intrinsics.cpu().numpy()
# points_2d = project_points_3d_to_2d(
#     points_3d=dataset_train.point_clouds[0],
#     intrinsics=intrinsics,
#     c2w=dataset_train.cameras[camera_idx].get_pose().cpu().numpy(),
# )
# # filter out points outside image range
# points_2d = points_2d[points_2d[:, 0] > 0]
# points_2d = points_2d[points_2d[:, 1] > 0]
# points_2d = points_2d[points_2d[:, 0] < img_np.shape[1]]
# points_2d = points_2d[points_2d[:, 1] < img_np.shape[0]]
# print("points_2d", points_2d.shape)

# fig = plt.figure()
# plt.imshow(img_np, alpha=1.0)
# colors = np.column_stack([points_2d, np.zeros((points_2d.shape[0], 1))])
# colors /= np.max(colors)
# colors += 0.5
# colors /= np.max(colors)
# plt.scatter(points_2d[:, 0], points_2d[:, 1], s=5, c=colors, marker=".")
# plt.gca().set_aspect("equal", adjustable="box")
# plt.xlabel("x")
# plt.ylabel("y")
