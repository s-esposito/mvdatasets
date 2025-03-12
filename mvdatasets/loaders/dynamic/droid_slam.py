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
    qw, qx, qy, qz = tum_pose[3:]  # Quaternion (qw, qx, qy, qz)
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
    config: dict,
    verbose: bool = False,
):
    """droid-slam data format loader.

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
    splits = config["splits"]

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

    # open poses.npy
    poses_file = scene_path / "poses.npy"
    poses = np.load(poses_file)  # (N, 8)
    # iterate over entries
    poses_list = []
    for pose in poses:
        poses_list.append(_tum_to_c2w(pose))
    print(f"Loaded poses.npy, {len(poses_list)} frames found")
        
    # open tstamps.npy
    tstamps_file = scene_path / "tstamps.npy"
    timestamps = np.load(tstamps_file)
    # convert timestamps to int
    frames_idxs = timestamps.astype(int)
    # print("frames_idxs", frames_idxs)
    timestamps = timestamps / config["frame_rate"]  # (N,)

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
    local_transform = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # load points.npy
    points_file = scene_path / "points.npy"
    frames_points = np.load(points_file)  # (N, 3)
    # print(frames_points.shape)
    point_clouds = []
    # for frame_points in frames_points:
    #     point_clouds.append(PointCloud(frame_points.reshape(-1, 3)))
        
    # rgbs_list = None
    # depths_list = None
    # if config["pose_only"]:
    #     # # skip loading rgb frames
    #     # # but load first one to get width and height
    #     # rgb = image_to_numpy(Image.open(rgb_frames[0]), use_uint8=True)
    #     # width, height = rgb.shape[1], rgb.shape[0]
    #     raise NotImplementedError("pose_only not implemented yet")
    # else:
    # load low resolution droid-slam images.npy
    images_file = scene_path / "images.npy"
    images_all = np.load(images_file)  # (N, 3, H, W)
    # reshape to (N, H, W, 3)
    images_all = np.moveaxis(images_all, 1, -1)
    # BGR to RGB
    images_all = images_all[..., ::-1]
    ds_width, ds_height = images_all.shape[2], images_all.shape[1]
    print(images_all.shape)
    
    # load low resolution droid-slam depths
    # if config["load_depths"]:
    # depths_file = scene_path / "disps.npy"
    # depths_all = np.load(depths_file)  # (N, H, W)  # TODO: divide for depths
    # print(depths_all.shape)
    
    # find all frame_0000.png frames rgb
    rgb_frames_paths = list(scene_path.glob("rgb/*.jpg"))
    rgb_frames_paths = sorted(rgb_frames_paths)
    print(f"Found {len(rgb_frames_paths)} RGB frames")
    
    # find all frame_0000.npy frames depth
    depth_frames_paths = list(scene_path.glob("depth/*.npy"))
    depth_frames_paths = sorted(depth_frames_paths)
    print(f"Found {len(depth_frames_paths)} depth frames")
    
    #
    rgb_frames = []
    pbar = tqdm(rgb_frames_paths, desc="rgbs", ncols=100)
    for rgb_frame_path in pbar:
        rgb = image_to_numpy(Image.open(rgb_frame_path), use_uint8=True)
        rgb_frames.append(rgb)
    
    # 
    depth_frames = []
    pbar = tqdm(depth_frames_paths, desc="depths", ncols=100)
    for depth_frame_path in pbar:
        depth = np.load(depth_frame_path)
        # multiply depth times scene scale mult
        depth *= scene_radius_mult
        depth_frames.append(depth)
        
    # get first image shape
    width, height = rgb_frames[0].shape[1], rgb_frames[0].shape[0]
    
    print(f"Loaded {len(rgb_frames)} RGB frames, {len(depth_frames)} depth frames")
    print(f"Image shape: {width}x{height}, droid-slam shape: {ds_width}x{ds_height}")
    
    # open intrinsics.npy
    pred_intrinsics_file = scene_path / "intrinsics.npy"
    intrinsics = np.load(pred_intrinsics_file)[0]
    fx = intrinsics[0]
    fy = intrinsics[1]
    cx = width / 2  # intrinsics[2]
    cy = height / 2  # intrinsics[3]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print(intrinsics)
    
    # TODO: do need to rescale focal length?

    # cameras objects
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []

        for i, idx in enumerate(frames_idxs):
            
            # if images_all is None:
            #     rgbs = None
            # else:
            rgbs = rgb_frames[idx][None, ...]
            
            # if depths_all is None:
            #     depths = None
            # else:
            depths = depth_frames[idx][None, ..., None]

            # create camera object
            camera = Camera(
                intrinsics=intrinsics,
                pose=poses_list[i],
                global_transform=global_transform,
                local_transform=local_transform,
                rgbs=rgbs,
                depths=depths,
                masks=None,
                timestamps=timestamps[i],
                camera_label=str(idx),
                width=width,
                height=height,
                subsample_factor=config["subsample_factor"],
                # verbose=verbose,
            )
            cameras_splits[split].append(camera)

    return {
        "scene_type": config["scene_type"],
        "point_clouds": point_clouds,
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_radius": scene_radius,
        "fps": config["frame_rate"],
        "nr_per_camera_frames": 1,
        "nr_sequence_frames": len(cameras_splits["train"]),
    }
