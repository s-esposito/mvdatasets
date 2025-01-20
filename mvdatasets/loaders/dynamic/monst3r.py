from rich import print
import numpy as np
from pathlib import Path
import os
import json
from PIL import Image
from tqdm import tqdm
from mvdatasets.utils.images import image_to_numpy
from mvdatasets import Camera
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.utils.printing import print_error, print_warning, print_success
from mvdatasets.geometry.common import (
    deg2rad,
    scale_3d,
    rot_x_3d,
    rot_y_3d,
    get_min_max_cameras_distances,
)
from mvdatasets.geometry.quaternions import quats_to_rots


def _tum_to_c2w(tum_pose):
    """
    Convert a TUM pose (x, y, z, qw, qx, qy, qz) back to a 4x4 c2w matrix.

    input: tum_pose: 1D array or list [x, y, z, qw, qx, qy, qz]
    output: c2w: 4x4 numpy array
    """
    # Extract translation and quaternion
    xyz = tum_pose[:3]  # Translation vector (x, y, z)
    qw, qx, qy, qz = tum_pose[3:-1]  # Quaternion (qw, qx, qy, qz)
    quat = np.array([qx, qy, qz, qw])

    # Convert quaternion to rotation matrix
    rotation_matrix = quats_to_rots(quat)

    # Construct the 4x4 transformation matrix
    c2w = np.eye(4)  # Start with identity matrix
    c2w[:3, :3] = rotation_matrix
    c2w[:3, 3] = xyz

    return c2w


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str] = ["train", "val"],
    config: dict = {},
    verbose: bool = False,
):
    """monst3r data format loader.

    Args:
        dataset_path (Path): Path to the dataset folder.
        scene_name (str): Name of the scene / sequence to load.
        splits (list): Splits to load (e.g., ["train", "val"]).
        config (dict): Dictionary of configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        cameras_splits (dict): Dictionary of splits with lists of Camera objects.
        global_transform (np.ndarray): (4, 4)
    """
    scene_path = dataset_path / scene_name

    # Update config with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            if verbose:
                print_warning(f"Setting '{key}' to default value: {default_value}")
        else:
            if verbose:
                print_success(f"Using '{key}': {config[key]}")

    # Valid values for specific keys
    valid_values = {
        "subsample_factor": [1],
    }

    # Validate specific keys
    for key, valid in valid_values.items():
        if key in config and config[key] not in valid:
            raise ValueError(f"{key} {config[key]} must be a value in {valid}")

    # Debugging output
    if verbose:
        print("load_monst3r config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    # open pred_traj.txt
    pred_traj_file = scene_path / "pred_traj.txt"
    # read all lines as np.array
    with open(pred_traj_file, "r") as f:
        lines = f.readlines()
    # iterate over lines
    poses_list = []
    tt_list = []
    for line in lines:
        tum_pose = np.array([float(x) for x in line.split()])  # (8,)
        tt = tum_pose[-1]
        # timestamp
        tt_list.append(tt)
        # convert to 4x4 c2w matrix
        c2w = _tum_to_c2w(tum_pose)
        # print(tt, c2w)
        poses_list.append(c2w)

    # open pred_intrinsics.txt
    pred_intrinsics_file = scene_path / "pred_intrinsics.txt"
    # read all lines as np.array
    with open(pred_intrinsics_file, "r") as f:
        lines = f.readlines()
    # iterate over lines
    intrinsics_list = []
    for line in lines:
        intrinsics = np.array([float(x) for x in line.split()])  # (9,)
        intrinsics = intrinsics.reshape(3, 3)
        intrinsics_list.append(intrinsics)

    # find scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(poses_list)

    if config["rescale"]:

        # scene scale such that furthest away camera is at target distance
        scene_radius_mult = config["target_max_camera_distance"] / max_camera_distance

        # new scene scale
        new_min_camera_distance = min_camera_distance * scene_radius_mult
        new_max_camera_distance = max_camera_distance * scene_radius_mult

        # scene radius
        scene_radius = new_max_camera_distance

    else:
        # scene scale such that furthest away camera is at target distance
        scene_radius_mult = 1.0

        # new scene scale
        new_min_camera_distance = min_camera_distance
        new_max_camera_distance = max_camera_distance

        # scene radius
        scene_radius = max_camera_distance

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    global_transform[:3, :3] = scene_radius_mult * rot_x_3d(
        deg2rad(rotate_scene_x_axis_deg)
    )

    # local transform
    local_transform = np.eye(4)

    # find all frame_0000.png frames rgb
    rgb_frames = list(scene_path.glob("frame_*.png"))
    rgb_frames = sorted(rgb_frames)
    print(f"Found {len(rgb_frames)} RGB frames")

    # find all frame_0028.npy frames depth
    depth_frames = list(scene_path.glob("frame_*.npy"))
    depth_frames = sorted(depth_frames)
    print(f"Found {len(depth_frames)} depth frames")

    # find all enlarged_dynamic_mask_0.png frames masks
    masks_frames = list(scene_path.glob("enlarged_dynamic_mask_*.png"))
    masks_frames = sorted(masks_frames)
    print(f"Found {len(masks_frames)} masks frames")

    rgbs_list = []
    depths_list = []
    masks_list = []
    if config["pose_only"]:
        # skip loading rgb frames
        # but load first one to get width and height
        rgb = image_to_numpy(Image.open(rgb_frames[0]), use_uint8=True)
        width, height = rgb.shape[1], rgb.shape[0]
    else:
        # load all rgb frames
        pbar = tqdm(rgb_frames, desc="rgbs", ncols=100)
        for rgb_frame in pbar:
            rgb = image_to_numpy(Image.open(rgb_frame), use_uint8=True)
            width, height = rgb.shape[1], rgb.shape[0]
            # print(rgb.shape, rgb.dtype)
            rgbs_list.append(rgb)

        # load all depth frames
        if config["load_depths"]:
            pbar = tqdm(depth_frames, desc="depths", ncols=100)
            for depth_frame in pbar:
                depth_np = np.load(depth_frame)  # (H, W)
                # multiply depth times scene scale mult
                depth_np *= scene_radius_mult
                # unsqueeze to (H, W, 1)
                depth_np = np.expand_dims(depth_np, axis=-1)
                # print(depth.shape, depth.dtype)
                depths_list.append(depth_np)

        if config["load_masks"]:
            # TODO
            pass

    # cameras objects
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []

        for i in range(len(rgbs_list)):
            # create camera object
            camera = Camera(
                intrinsics=intrinsics_list[i],
                pose=poses_list[i],
                global_transform=global_transform,
                local_transform=local_transform,
                rgbs=rgbs_list[i][None, ...],
                depths=depths_list[i][None, ...],
                masks=None,
                timestamps=tt_list[i],
                camera_label=str(i),
                width=width,
                height=height,
                subsample_factor=config["subsample_factor"],
                # verbose=verbose,
            )
            cameras_splits[split].append(camera)

    return {
        "scene_type": config["scene_type"],
        "point_clouds": [],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "min_camera_distance": new_min_camera_distance,
        "max_camera_distance": new_max_camera_distance,
        "scene_radius": scene_radius,
        "fps": config["frame_rate"],
        "nr_per_camera_frames": 1,
        "nr_sequence_frames": len(cameras_splits["train"]),
    }
