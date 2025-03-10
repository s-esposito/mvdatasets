import numpy as np
import PIL
import os
import sys
import time
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import imageio
import json

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.utils.data_converter import convert_to_ingp_format


if __name__ == "__main__":

    # Set a random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(SEED)  # Set a random seed for GPU
    else:
        device = "cpu"
    torch.set_default_device(DEVICE)

    # Set default tensor type
    torch.set_default_dtype(torch.float32)

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code

    # Get dataset test preset
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "dtu"
    scene_name, pc_paths, config = get_dataset_test_preset(args.dataset_name)

    # dataset loading
    mv_data = MVDataset(
        args.dataset_name,
        scene_name,
        args.datasets_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True,
    )

    scene = {}
    scene["resolution"] = [mv_data.get_width(), mv_data.get_height()]
    scene["meshes"] = []
    scene["cameras"] = {"test": {}, "train": {}}
    for camera in mv_data.get_split("test"):
        camera_idx = camera.camera_label
        projectionMatrix = camera.get_opengl_projection_matrix()
        matrixWorld = camera.get_opengl_matrix_world()
        scene["cameras"]["test"][camera_idx] = {
            "projectionMatrix": projectionMatrix.tolist(),
            "matrixWorld": matrixWorld.tolist(),  # c2w
        }

    # Save the projections as a json file
    with open("out/scene.json", "w") as f:
        json.dump(scene, f, indent=4)
