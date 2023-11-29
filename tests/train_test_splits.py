import numpy
import PIL
import os
import sys
import time
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import imageio

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.utils.plotting import (
    plot_cameras,
    plot_camera_rays,
    plot_current_batch,
    plot_points_2d_on_image,
)
from mvdatasets.utils.raycasting import get_pixels, get_camera_rays, get_camera_frames
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.geometry import project_points_3d_to_2d

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# # Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(seed)  # Set a random seed for GPU
else:
    device = "cuda"
torch.set_default_device(device)

# Set default tensor type
torch.set_default_dtype(torch.float32)

# Set profiler
profiler = Profiler()  # nb: might slow down the code

datasets_path = "/home/stefano/Data"
dataset_names = ["dtu", "blender"]
scene_names = ["dtu_scan83", "lego"]

for dataset_name, scene_name in zip(dataset_names, scene_names):

    # load gt mesh if exists
    gt_meshes_paths = [os.path.join("debug/meshes/", dataset_name, scene_name, "mesh.ply")]

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=gt_meshes_paths,
        splits=["train", "test"]
    )

    # Visualize cameras
    fig = plot_cameras(
        mv_data["train"],
        points=mv_data.point_clouds[0],
        azimuth_deg=20,
        elevation_deg=30,
        up="y",
        figsize=(15, 15),
        title="training cameras",
    )

    #  plt.show()
    plt.savefig(os.path.join("imgs", f"{dataset_name}_training_cameras.png"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    # Visualize cameras
    fig = plot_cameras(
        mv_data["test"],
        points=mv_data.point_clouds[0],
        azimuth_deg=20,
        elevation_deg=30,
        up="y",
        figsize=(15, 15),
        title="test cameras",
    )

    #  plt.show()
    plt.savefig(os.path.join("imgs", f"{dataset_name}_test_cameras.png"), transparent=True, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()