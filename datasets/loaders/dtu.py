import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2 as cv

from datasets.utils.images import numpy2image, image2numpy
from datasets.scenes.camera import Camera
from datasets.utils.geometry import (
    deg2rad,
    scale_3d,
    rot_x_3d,
    rot_y_3d,
    pose_local_rotation,
    pose_global_rotation,
)


# from https://github.com/Totoro97/NeuS/blob/main/models/dataset.py
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def load_dtu(
    data_path,
    load_with_mask=True,
    rotate_scene_x_axis_degrees=115,
    scene_scale_multiplier=0.4,
    # downscale_factor=1
    device="cpu",
):
    # create cameras objects
    cameras = []

    # load images to cpu as numpy arrays
    imgs = []
    images_list = sorted(glob(os.path.join(data_path, "image/*.png")))
    pbar = tqdm(images_list, desc="Loading images", leave=True)
    for im_name in pbar:
        # load PIL image
        img_pil = Image.open(im_name)
        img_np = image2numpy(img_pil)
        # img_np = img_np[::downscale_factor, ::downscale_factor]
        imgs.append(img_np)

    # (optional) load mask images to cpu as numpy arrays
    masks = []
    if load_with_mask:
        masks_list = sorted(glob(os.path.join(data_path, "mask/*.png")))
        pbar = tqdm(masks_list, desc="Loading masks", leave=True)
        for im_name in pbar:
            # load PIL image
            mask_pil = Image.open(im_name)
            mask_np = image2numpy(mask_pil)
            mask_np = mask_np[:, :, 0, None]
            # mask_np = mask_np[::downscale_factor, ::downscale_factor]
            masks.append(mask_np)

    camera_dict = np.load(os.path.join(data_path, "cameras_sphere.npz"))
    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict[f"world_mat_{idx}"] for idx in range(len(images_list))]
    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [
        camera_dict["scale_mat_%d" % idx] for idx in range(len(images_list))
    ]

    # decompose into intrinsics and extrinsics
    for idx, mats in enumerate(zip(world_mats_np, scale_mats_np)):
        world_mat_np, scale_mat_np = mats

        projection_np = world_mat_np @ scale_mat_np
        projection_np = projection_np[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, projection_np)

        # flip local x-axis
        # pose[:3, 0] *= -1

        # rotation around x axis by 115 degrees
        rotation = rot_x_3d(deg2rad(rotate_scene_x_axis_degrees))
        pose = pose_global_rotation(pose, rotation)

        # transformation flipping the z-axis
        # rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        # pose = pose_global_rotation(pose, rotation)

        # rotation around x axis by 115 degrees
        # rotation = rot_x_3d(deg2rad(-rotate_scene_x_axis_degrees))
        # pose = pose_global_rotation(pose, rotation)

        pose[:3, 0] *= -1
        pose[:3, 2] *= -1
        pose[:3, 1] *= -1

        # scale
        pose[:3, 3] *= scene_scale_multiplier

        # scale = scale_3d(scene_scale_multiplier)
        # tf_world_cam_rescaled=recaling_matrix*tf_world_cam_rescaled;
        # tf_cam_world=tf_world_cam_rescaled.inverse();

        # rotates pose by 90 degrees around world x axis
        # pose = pose_global_rotation(pose, rot_x_3d(np.pi / 2))

        # rotate local frame by 180 degrees around y axis
        # pose = pose_local_rotation(pose, rot_y_3d(np.pi))

        # get images
        cam_imgs = imgs[idx][None, ...]

        # get mask (optional)
        if load_with_mask and len(masks) > idx:
            cam_masks = masks[idx][None, ...]
        else:
            cam_masks = None

        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
            imgs=cam_imgs,
            masks=cam_masks,
            device=device,
        )

        cameras.append(camera)

    return cameras
