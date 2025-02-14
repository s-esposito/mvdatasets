from rich import print
from pathlib import Path
import os
import numpy as np
from pycolmap import SceneManager
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

from mvdatasets import Camera
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.utils.loader_utils import rescale
from mvdatasets.geometry.common import rot_euler_3d_deg
from mvdatasets.utils.printing import print_warning


def load(
    dataset_path: Path,
    scene_name: str,
    config: dict,
    verbose: bool = False,
):
    """TUM data format loader.

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
    valid_values = {}

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
    
    # get all rgbs paths
    rgbs_dir = scene_path / "rgb"
    rgbs_paths = sorted(list(rgbs_dir.glob("*.png")))
    # sort by name
    rgbs_paths = sorted(rgbs_paths, key=lambda x: "".join(str(x).split(".")[:-1]))
    
    # get all depth paths
    depths_dir = scene_path / "depth"
    depths_paths = sorted(list(depths_dir.glob("*.png")))
    # sort by name
    depths_paths = sorted(depths_paths, key=lambda x: "".join(str(x).split(".")[:-1]))
    
    # parse groundtruth.txt
    # timestamp tx ty tz qx qy qz qw
    poses = []
    with open(scene_path / "groundtruth.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip().split(" ")
            timestamp = int(line[0])
            tx, ty, tz = map(float, line[1:4])
            qx, qy, qz, qw = map(float, line[4:])
            poses.append([timestamp, tx, ty, tz, qx, qy, qz, qw])
    
    frame_rate = 30
    
    raise NotImplementedError("tum loader not implemented yet")