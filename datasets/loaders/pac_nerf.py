import json
import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
import open3d as o3d
from PIL import Image

from datasets.utils.images import numpy2image, image2numpy
from datasets.utils.geometry import rot_x_3d, rot_z_3d, pose_local_rotation
from datasets.scenes.camera import Camera

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


# def load_particles_pacnerf(path, nr_points=1000):
#     # read point cloud from ply file
#     point_cloud = o3d.io.read_point_cloud(path)
#     points = np.asarray(point_cloud.points)
#     # downsample selecting nr_points random points
#     random_idx = np.random.choice(points.shape[0], nr_points, replace=False)
#     points = points[random_idx]
#     return points


def load_pac_nerf(data_path, n_cameras=1, load_with_mask=False, device="cpu"):
    """
    Load pac_nerf data.
    """

    with open(os.path.join(data_path, "all_data.json"), encoding="utf-8") as f:
        data_info = json.load(f)

    n_frames = int(len(data_info) / n_cameras) - 1

    poses_all = np.zeros((n_cameras, 4, 4))
    intrinsics_all = np.zeros((n_cameras, 3, 3))
    imgs_all = None
    bg_imgs_all = None

    for entry in tqdm(data_info):
        cam_id, frame_id = [
            int(i)
            for i in entry["file_path"]
            .split("/")[-1]
            .rstrip(".png")
            .lstrip("r_")
            .split("_")
        ]

        poses_all[cam_id] = np.eye(4)

        pose = np.eye(4)
        pose[:3, :4] = np.array(entry["c2w"])
        # invert z axis
        # pose[:3, 2] *= -1
        # rotate 180 around z-axis
        # rot_z = rot_z_3d(np.pi)
        # pose = pose_local_rotation(pose, rot_z)
        # invert x axis
        pose[:3, 0] *= -1
        # pose = pose_local_rotation(pose, rot_x)

        poses_all[cam_id] = pose
        intrinsics_all[cam_id] = entry["intrinsic"]
        img_pil = Image.open(os.path.join(data_path, entry["file_path"]))
        img_np = image2numpy(img_pil)[..., :3]

        if frame_id < 0:
            # background frame
            if bg_imgs_all is None:
                # need to read image dimensions first
                height, width = img_np.shape[:2]
                bg_imgs_all = np.zeros((n_cameras, height, width, 3))
            bg_imgs_all[cam_id] = img_np

        if imgs_all is None:
            # need to read image dimensions first
            height, width = img_np.shape[:2]
            imgs_all = np.zeros((n_cameras, n_frames, height, width, 3))

        imgs_all[cam_id, frame_id] = img_np

    # nb: not working great, image differences are noisy
    # should use the same approach as in the original code
    masks_all = np.zeros((n_cameras, n_frames, height, width, 1))
    if load_with_mask:
        # use background image to construct per frame masks
        for cam_id in range(n_cameras):
            for frame_id in range(n_frames):
                masks_all[cam_id, frame_id] = np.all(
                    imgs_all[cam_id, frame_id] == bg_imgs_all[cam_id],
                    axis=-1,
                    keepdims=True,
                ).astype(np.float32)

    cameras = []
    if load_with_mask:
        for intrinsics, pose, imgs, masks in zip(
            intrinsics_all, poses_all, imgs_all, masks_all
        ):
            cameras.append(
                Camera(
                    imgs, masks=masks, intrinsics=intrinsics, pose=pose, device=device
                )
            )
    else:
        for intrinsics, pose, imgs in zip(intrinsics_all, poses_all, imgs_all):
            cameras.append(
                Camera(imgs, intrinsics=intrinsics, pose=pose, device=device)
            )

    return cameras
