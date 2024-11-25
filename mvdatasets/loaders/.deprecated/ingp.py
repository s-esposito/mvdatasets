from rich import print
from pathlib import Path
import os
import json
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from mvdatasets import Camera
from mvdatasets.utils.images import image_to_numpy
from mvdatasets.geometry.common import (
    deg2rad,
    scale_3d,
    rot_x_3d,
    rot_y_3d,
    pose_local_rotation,
    pose_global_rotation,
    get_min_max_cameras_distances,
)
from mvdatasets.utils.images import image_uint8_to_float32, image_float32_to_uint8
from mvdatasets.utils.printing import print_error, print_warning


def load_ingp(dataset_path: Path, scene_name: str, splits: list, config: dict = {}, verbose: bool = False):
    """INGP data format loader.

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
    # Default configuration
    defaults = {
        "scene_type": "bounded",
        "rotate_scene_x_axis_deg": 0.0,
        "subsample_factor": 1,
        "test_camera_freq": 8,
        "train_test_overlap": False,
        "target_max_camera_distance": 1.0,
        "init_sphere_radius_mult": 0.5,
        "pose_only": False,
    }

    # Update config with defaults and handle warnings
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            if verbose:
                print_warning(f"{key} not in config, setting to {default_value}")

    # Check for unimplemented features
    if config.get("pose_only"):
        if verbose:
            print_warning("pose_only is True, but this is not implemented yet")

    # Debugging output
    if verbose:
        print("load_ingp config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    # load camera params
    with open(os.path.join(scene_path, "transforms.json"), "r") as fp:
        metas = json.load(fp)

    # height, width = metas["h"], metas["w"]
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = metas["fl_x"]
    intrinsics[1, 1] = metas["fl_y"]
    intrinsics[0, 2] = metas["cx"]
    intrinsics[1, 2] = metas["cy"]

    # read all poses
    poses_all = []
    for frame in metas["frames"]:
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        poses_all.append(pose)

    # find scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(poses_all)

    # define scene scale
    scene_scale = max_camera_distance  # (1/metas["aabb_scale"])
    # round to 2 decimals
    scene_scale = round(scene_scale, 2)

    # scene scale such that furthest away camera is at target distance
    scene_scale_mult = config["target_max_camera_distance"] / (
        max_camera_distance + 1e-2
    )

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    global_transform[:3, :3] = scene_scale_mult * rot_x_3d(
        deg2rad(rotate_scene_x_axis_deg)
    )

    # local transform
    local_transform = np.eye(4)
    local_transform[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # cameras objects
    cameras_all = []
    for i, frame in enumerate(metas["frames"]):

        pose = poses_all[i]
        img_path = os.path.join(scene_path, frame["file_path"])

        # check if file exists
        if not os.path.exists(img_path):
            print(f"[bold yellow]WARNING[/bold yellow]: {img_path} does not exist")
            continue

        idx = int(img_path.split("/")[-1].split(".")[0])

        # load PIL image
        img_pil = Image.open(img_path)
        img_np = image_to_numpy(img_pil, use_uint8=True)[:, :, :3]

        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=img_np[None, ...],
            masks=None,
            camera_idx=idx,
            subsample_factor=int(config["subsample_factor"]),
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
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "config": config,
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_scale": scene_scale,
        "scene_scale_mult": scene_scale_mult,
    }
