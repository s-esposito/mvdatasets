from rich import print
from pathlib import Path
import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2 as cv

from mvdatasets.utils.images import image_to_numpy
from mvdatasets import Camera
from mvdatasets.geometry.common import (
    deg2rad,
    scale_3d,
    rot_x_3d,
    rot_y_3d,
    get_min_max_cameras_distances,
)
from mvdatasets.utils.printing import print_error, print_warning, print_success


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


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str] = ["train", "test"],
    config: dict = {},
    verbose: bool = False,
):
    """DTU data format loader.

    Args:
        dataset_path (Path): Path to the dataset folder.
        scene_name (str): Name of the scene / sequence to load.
        splits (list): Splits to load (e.g., ["train", "test"]).
        config (dict): Dictionary of configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        cameras_splits (dict): Dictionary of splits with lists of Camera objects.
        global_transform (np.ndarray): (4, 4)
    """

    scene_path = dataset_path / scene_name

    # Default configuration
    defaults = {
        # "scene_type": "unbounded",
        # "load_masks": True,
        # "test_camera_freq": 8,
        # "train_test_overlap": False,
        # "rotate_scene_x_axis_deg": 205,
        # "subsample_factor": 1,
        # "init_sphere_radius_mult": 0.1,
        # "target_max_camera_distance": 0.5,
        # "foreground_scale_mult": 1.0,
        # "pose_only": False,
    }

    # Update config with defaults and handle warnings
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            if verbose:
                print_warning(f"{key} not in config, setting to {default_value}")
        else:
            if verbose:
                print_success(f"Using '{key}': {config[key]}")

    # Debugging output
    if verbose:
        print("load_dtu config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    # load images to cpu as numpy arrays
    imgs = []
    masks = []

    if not config["pose_only"]:

        images_list = sorted(glob(os.path.join(scene_path, "image/*.png")))
        pbar = tqdm(images_list, desc="images", ncols=100)
        for im_name in pbar:
            # load PIL image
            img_pil = Image.open(im_name)
            img_np = image_to_numpy(img_pil, use_uint8=True)
            imgs.append(img_np)

        # (optional) load mask images to cpu as numpy arrays
        if config["load_masks"]:
            masks_list = sorted(glob(os.path.join(scene_path, "mask/*.png")))
            pbar = tqdm(masks_list, desc="masks", ncols=100)
            for im_name in pbar:
                # load PIL image
                mask_pil = Image.open(im_name)
                mask_np = image_to_numpy(mask_pil, use_uint8=True)
                mask_np = mask_np[:, :, 0, None]
                masks.append(mask_np)

    # load camera params
    camera_dict = np.load(os.path.join(scene_path, "cameras_sphere.npz"))
    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict[f"world_mat_{idx}"] for idx in range(len(images_list))]
    # scale_mat: used for coordinate normalization,
    # we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [
        camera_dict["scale_mat_%d" % idx] for idx in range(len(images_list))
    ]

    # decompose into intrinsics and extrinsics
    intrinsics_all = []
    poses_all = []
    for idx, mats in enumerate(zip(world_mats_np, scale_mats_np)):
        world_mat_np, scale_mat_np = mats

        projection_np = world_mat_np @ scale_mat_np
        projection_np = projection_np[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, projection_np)

        intrinsics_all.append(intrinsics)
        poses_all.append(pose)

    # find original scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(poses_all)

    # scene scale such that furthest away camera is at target distance
    scene_radius_mult = config["target_max_camera_distance"] / max_camera_distance

    # new scene scale
    new_min_camera_distance = min_camera_distance * scene_radius_mult
    new_max_camera_distance = max_camera_distance * scene_radius_mult

    # scene radius
    scene_radius = new_max_camera_distance

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    global_transform[:3, :3] = scene_radius_mult * rot_x_3d(
        deg2rad(rotate_scene_x_axis_deg)
    )

    # local transform
    local_transform = np.eye(4)
    local_transform[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    cameras_all = []
    for idx, params in enumerate(zip(intrinsics_all, poses_all)):

        intrinsics, pose = params

        # get images
        if len(imgs) > idx:
            cam_imgs = imgs[idx][None, ...]
        else:
            cam_imgs = None

        # get mask (optional)
        if len(masks) > idx:
            cam_masks = masks[idx][None, ...]
        else:
            cam_masks = None

        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=cam_imgs,
            masks=cam_masks,
            camera_label=str(idx),
            subsample_factor=int(config["subsample_factor"]),
            # verbose=verbose,
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

    return {
        "scene_type": config["scene_type"],
        "init_sphere_radius_mult": config["init_sphere_radius_mult"],
        "foreground_scale_mult": config["foreground_scale_mult"],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "min_camera_distance": new_min_camera_distance,
        "max_camera_distance": new_max_camera_distance,
        "scene_radius": scene_radius,
    }
