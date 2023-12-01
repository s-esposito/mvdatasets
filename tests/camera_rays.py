import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.utils.plotting import plot_camera_rays
from mvdatasets.utils.raycasting import get_camera_rays, get_camera_frames
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler

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

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        splits=["train", "test"],
    )

    # random camera index
    rand_idx = torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])
    print(camera)

    # resize camera's rgb modality
    camera.resize(max_dim=100)

    # gen rays
    rays_o, rays_d, points_2d = get_camera_rays(camera)
    # print("rays_o", rays_o.shape, rays_o.device)
    # print("rays_d", rays_d.shape, rays_d.device)
    # print("points_2d", points_2d.shape, points_2d.device)

    vals, _ = get_camera_frames(camera, points_2d=points_2d)
    for key, val in vals.items():
        print(key, val.shape, val.device)

    # visualize camera
    fig = plot_camera_rays(
        camera, 512, azimuth_deg=60, elevation_deg=30, up="y", figsize=(15, 15)
    )

    # plt.show()
    plt.savefig(
        os.path.join("imgs", f"{dataset_name}_camera_rays.png"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True
    )
    plt.close()
    
    # # plt.show()
    # img_path = os.path.join("plots", f"{dataset_name}_camera_test_{rand_idx}.png")
    # img = camera.get_rgb()
    # mask = camera.get_mask()
    
    # # concatenate mask 3 times
    # mask = np.concatenate([mask] * 3, axis=-1)
    # print("mask", mask.shape)
    
    # # save image
    # plt.imsave(img_path, img)
    # plt.imsave(img_path.replace(".png", "_mask.png"), mask)