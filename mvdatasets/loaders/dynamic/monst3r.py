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
from mvdatasets.utils.loader_utils import rescale
from mvdatasets.geometry.common import rot_euler_3d_deg
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
    splits: list[str],
    config: dict,
    verbose: bool = False,
):
    """monst3r data format loader.

    Args:
        dataset_path (Path): Path to the dataset folder.
        scene_name (str): Name of the scene / sequence to load.
        splits (list): Splits to load (e.g., ["train", "val"]).
        config (DatasetConfig): Dataset configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        dict: Dictionary of splits with lists of Camera objects.
        np.ndarray: Global transform (4, 4)
        str: Scene type
        List[PointCloud]: List of PointClouds
        float: Minimum camera distance
        float: Maximum camera distance
        float: Foreground scale multiplier
        float: Scene radius
        int: Number of frames per camera
        int: Number of sequence frames
        float: Frames per second
    """

    scene_path = dataset_path / scene_name

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
        print("config:")
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

    # rescale (optional)
    scene_radius_mult, min_camera_distance, max_camera_distance = rescale(
        poses_list, to_distance=config["max_cameras_distance"]
    )
    
    scene_radius = max_camera_distance

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rot = rot_euler_3d_deg(
        config["rotate_deg"][0], config["rotate_deg"][1], config["rotate_deg"][2]
    )
    global_transform[:3, :3] = scene_radius_mult * rot

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
    masks_frames = sorted(masks_frames, key=lambda x: int(x.stem.split("_")[-1]))
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

        # load all mask frames
        if config["load_masks"]:
            pbar = tqdm(masks_frames, desc="masks", ncols=100)
            for mask_frame in pbar:
                mask_np = image_to_numpy(Image.open(mask_frame), use_uint8=True)
                mask_np = mask_np[:, :, None]
                masks_list.append(mask_np)

    # cameras objects
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []

        for i in range(len(rgbs_list)):
            
            if len(depths_list) == 0:
                depths = None
            else:
                depths = depths_list[i][None, ...]
            
            if len(masks_list) == 0:
                masks = None
            else:
                masks = masks_list[i][None, ...]
            
            # create camera object
            camera = Camera(
                intrinsics=intrinsics_list[i],
                pose=poses_list[i],
                global_transform=global_transform,
                local_transform=local_transform,
                rgbs=rgbs_list[i][None, ...],
                depths=depths,
                masks=masks,
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
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_radius": scene_radius,
        "fps": config["frame_rate"],
        "nr_per_camera_frames": 1,
        "nr_sequence_frames": len(cameras_splits["train"]),
    }
