import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2 as cv

from mvdatasets.utils.images import image2numpy
from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.geometry import (
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
    splits,
    load_mask=True,
    test_camera_freq=8,
    train_test_overlap=False,
    rotate_scene_x_axis_deg=115,
    scene_scale_mult=0.4,
    subsample_factor=1,
):
    # cameras objects
    cameras_all = []
    
    # global transform
    global_transform = np.eye(4)
    # rotate
    rotation = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))
    # scale
    s_rotation = scene_scale_mult * rotation
    global_transform[:3, :3] = s_rotation
    
    # load images to cpu as numpy arrays
    imgs = []
    images_list = sorted(glob(os.path.join(data_path, "image/*.png")))
    pbar = tqdm(images_list, desc="images", ncols=100)
    for im_name in pbar:
        # load PIL image
        img_pil = Image.open(im_name)
        img_np = image2numpy(img_pil)
        imgs.append(img_np)

    # (optional) load mask images to cpu as numpy arrays
    masks = []
    if load_mask:
        masks_list = sorted(glob(os.path.join(data_path, "mask/*.png")))
        pbar = tqdm(masks_list, desc="masks", ncols=100)
        for im_name in pbar:
            # load PIL image
            mask_pil = Image.open(im_name)
            mask_np = image2numpy(mask_pil)
            mask_np = mask_np[:, :, 0, None]
            masks.append(mask_np)
    
    # load camera params
    camera_dict = np.load(os.path.join(data_path, "cameras_sphere.npz"))
    # world_mat is a projection matrix from world to image
    world_mats_np = [
        camera_dict[f"world_mat_{idx}"] for idx in range(len(images_list))
    ]
    # scale_mat: used for coordinate normalization,
    # we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [
        camera_dict["scale_mat_%d" % idx] for idx in range(len(images_list))
    ]

    # decompose into intrinsics and extrinsics
    for idx, mats in enumerate(zip(world_mats_np, scale_mats_np)):
        world_mat_np, scale_mat_np = mats

        projection_np = world_mat_np @ scale_mat_np
        projection_np = projection_np[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, projection_np)

        # get images
        cam_imgs = imgs[idx][None, ...]

        # get mask (optional)
        if load_mask and len(masks) > idx:
            cam_masks = masks[idx][None, ...]
        else:
            cam_masks = None

        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
            global_transform=global_transform,
            rgbs=cam_imgs,
            masks=cam_masks,
            camera_idx=idx,
        )

        cameras_all.append(camera)

    # split cameras into train and test
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []
        if split == "train":
            if train_test_overlap:
                # if train_test_overlap, use all cameras for training
                cameras_splits[split] = cameras_all
            # else use only a subset of cameras
            else:
                for i, camera in enumerate(cameras_all):
                    if i % test_camera_freq != 0:
                        cameras_splits[split].append(camera)
        if split == "test":
            # select a test camera every test_camera_freq cameras
            for i, camera in enumerate(cameras_all):
                if i % test_camera_freq == 0:
                    cameras_splits[split].append(camera)

    return cameras_splits, global_transform
