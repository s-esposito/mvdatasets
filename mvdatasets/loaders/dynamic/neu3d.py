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
    
    # Default configuration
    defaults = {
        "scene_type": "unbounded",
        "load_masks": True,
        "rescale": True,  # TODO: testing
        "rotate_scene_x_axis_deg": 0.0,
        "subsample_factor": 1,
        "target_max_camera_distance": 1.0,
        "foreground_scale_mult": 1.0,
        "frame_rate": 30.0,
        "pose_only": False,
    }

    # Update config with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            if verbose:
                print_warning(f"Setting '{key}' to default value: {default_value}")
        else:
            if verbose:
                print_success(f"Using '{key}': {config[key]}")

    # Debugging output
    if verbose:
        print("load_neu3d config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------
    
    # load poses
    # poses_bounds.npy
    poses_bounds_path = scene_path / "poses_bounds.npy"
    if not poses_bounds_path.exists():
        print_error(f"File {poses_bounds_path} does not exist.")
    
    poses_bounds = np.load(poses_bounds_path)
    print(poses_bounds)
    
    # camera poses are in OpenGL format
    # Poses are stored as 3x4 numpy arrays that represent camera-to-world 
    # transformation matrices. The other data you will need is simple pinhole 
    # camera intrinsics (hwf = [height, width, focal length]) and near/far scene bounds

    poses_list = []
    intrinsics_list = []
    for poses_bound in poses_bounds:
        c2w_ogl = poses_bound[:3*4].reshape(3, 4)  # :12
        height = poses_bound[3*4]  # 12
        width = poses_bound[3*4+1]  # 13
        focal_length = poses_bound[3*4+2]  # 14
        near = poses_bound[3*4+3]  # 15
        far = poses_bound[3*4+4]  # 16
        print("c2w", c2w_ogl)
        print("height", height)
        print("width", width)
        print("focal_length", focal_length)
        print("near", near)
        print("far", far)
        # poses_list.append(poses_bound[:3, :4])
        # intrinsics_list.append(poses_bound[3:])
    
    # find all cam00.mp4 cameras feeds
    cam_videos = list(scene_path.glob("cam*.mp4"))
    cam_videos = sorted(cam_videos)
    
    # check if frames where already extracted
    for cam_video in cam_videos:
        video_path = scene_path / cam_video
        frames_folder = scene_path / cam_video.stem
        if not frames_folder.exists():
            print_warning(f"Frames folder {frames_folder} does not exist.")
            # extract frames
            from mvdatasets.utils.video_utils import extract_frames
            
            extract_frames(
                video_path=video_path,
                subsample_factor=1,
                ext="jpg",
            )
            print_success(f"Frames extracted to {frames_folder}")
            
    exit(0)