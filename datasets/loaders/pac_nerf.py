import json
import torch
import numpy as np
import os
from tqdm import tqdm
import imageio
import cv2 as cv
import open3d as o3d

from datasets.utils.camera import Camera

# def load_particles_fluidsym(path):
#     f = open(path, "rb")
#     xyz = struct.unpack("iii", f.read(4 * 3))
#     num_part = struct.unpack("i", f.read(4))[0]
#     vals = np.array(struct.unpack("f" * 8 * num_part, f.read(8 * 4 * num_part))).reshape(num_part, 8)
#     points_3d, vel, nu, m = vals[:, :3], vals[:, 3:6], vals[:, 6:7], vals[:, 7:]

#     max_dim = max(xyz)
#     points_3d /= max_dim / 2

#     shift = np.array([0.0, 0.0, 1.0])
#     points_3d += shift

#     # world axis transform
#     axis_rot = rot_x_3d(-np.pi / 2)
#     axis_transform = np.eye(4)
#     axis_transform[:3, :3] = axis_rot
#     points_3d = transform_points_3d(points_3d, axis_transform)

#     return points_3d


def load_particles_pacnerf(path, nr_points=1000):
    # read point cloud from ply file
    point_cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(point_cloud.points)
    # downsample selecting nr_points random points
    random_idx = np.random.choice(points.shape[0], nr_points, replace=False)
    points = points[random_idx]
    return points


def load_pac_nerf(data_path, n_cameras=1, load_with_mask=False, device="cpu"):
    """
    Load pac_nerf data.
    """

    with open(os.path.join(data_path, "all_data.json"), encoding="utf-8") as f:
        data_info = json.load(f)

    n_frames = int(len(data_info) / n_cameras) - 1

    poses_all = np.zeros((n_cameras, 4, 4))
    intrinsics_all = np.zeros((n_cameras, 3, 3))
    rgb_all = None

    for entry in tqdm(data_info):
        cam_id, frame_id = [int(i) for i in entry["file_path"].split("/")[-1].rstrip(".png").lstrip("r_").split("_")]

        if frame_id < 0:
            # TODO: if load_with_mask is set to True
            # use background image to construct per frame masks
            continue

        poses_all[cam_id] = np.eye(4)
        poses_all[cam_id, :3, :4] = entry["c2w"]
        intrinsics_all[cam_id] = entry["intrinsic"]
        img = np.array(imageio.imread(os.path.join(data_path, entry["file_path"])))[..., :3]

        if rgb_all is None:
            # need to read image dimensions first
            height, width = img.shape[:2]
            rgb_all = np.zeros((n_cameras, n_frames, height, width, 3))

        rgb_all[cam_id, frame_id] = img

    cameras = []
    for intrinsics, pose, imgs in zip(intrinsics_all, poses_all, rgb_all):
        cameras.append(Camera(imgs, intrinsics=intrinsics, pose=pose, device=device))

    return cameras
