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
    scene_path,
    splits,
    config,
    verbose=False,
):
    """dtu data format loader

    Args:
        scene_path (str): path to the dataset scene folder
        splits (list): splits to load (e.g. ["train", "test"])
        config (dict): dict of config parameters

    Returns:
        cameras_splits (dict): dict of splits with lists of Camera objects
        global_transform (np.ndarray): (4, 4)
    """
    
    # CONFIG -----------------------------------------------------------------
    
    if "load_mask" not in config:
        if verbose:
            print("WARNING: load_mask not in config, setting to True")
        config["load_mask"] = True
    
    if "test_camera_freq" not in config:
        if verbose:
            print("WARNING: test_camera_freq not in config, setting to 8")
        config["test_camera_freq"] = 8
    
    if "train_test_overlap" not in config:
        if verbose:
            print("WARNING: train_test_overlap not in config, setting to False")
        config["train_test_overlap"] = False
        
    if "rotate_scene_x_axis_deg" not in config:
        if verbose:
            print("WARNING: rotate_scene_x_axis_deg not in config, setting to 115")
        config["rotate_scene_x_axis_deg"] = 115
    
    if "scene_scale_mult" not in config:
        if verbose:
            print("WARNING: scene_scale_mult not in config, setting to 0.4")
        config["scene_scale_mult"] = 0.4
    
    # TODO: implement subsample_factor
    # if "subsample_factor" not in config:
    #     config["subsample_factor"] = 1.0
    
    if verbose:
        print("dtu config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")
    
    # -------------------------------------------------------------------------
    
    # cameras objects
    cameras_all = []
    
    # global transform
    global_transform = np.eye(4)
    # rotate
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    rotation = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))
    # scale
    scene_scale_mult = config["scene_scale_mult"]
    s_rotation = scene_scale_mult * rotation
    global_transform[:3, :3] = s_rotation
    
    # load images to cpu as numpy arrays
    imgs = []
    images_list = sorted(glob(os.path.join(scene_path, "image/*.png")))
    pbar = tqdm(images_list, desc="images", ncols=100)
    for im_name in pbar:
        # load PIL image
        img_pil = Image.open(im_name)
        img_np = image2numpy(img_pil)
        imgs.append(img_np)

    # (optional) load mask images to cpu as numpy arrays
    masks = []
    if config["load_mask"]:
        masks_list = sorted(glob(os.path.join(scene_path, "mask/*.png")))
        pbar = tqdm(masks_list, desc="masks", ncols=100)
        for im_name in pbar:
            # load PIL image
            mask_pil = Image.open(im_name)
            mask_np = image2numpy(mask_pil)
            mask_np = mask_np[:, :, 0, None]
            masks.append(mask_np)
    
    # load camera params
    camera_dict = np.load(os.path.join(scene_path, "cameras_sphere.npz"))
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
        if config["load_mask"] and len(masks) > idx:
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
    train_test_overlap = config["train_test_overlap"]
    test_camera_freq = config["test_camera_freq"]
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
