import numpy
import PIL
import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

from datasets.utils.plotting import plot_cameras, plot_camera_rays, plot_current_batch
from datasets.mv_dataset import MVDataset
from datasets.utils.profiler import Profiler
from torch.utils.data import DataLoader

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(seed)  # Set a random seed for GPU
else:
    device = "cpu"
torch.set_default_device(device)

# Set default tensor type
torch.set_default_dtype(torch.float32)

# Set profiler
profiler = Profiler()  # nb: might slow down the code

data_path = "/home/stefano/Data"
dataset_name = "dtu"
scene_name = "dtu_scan83"

scene_data_path = os.path.join(data_path, dataset_name, scene_name)
# make sure folder exists
assert os.path.exists(scene_data_path), "Scene data path does not exist"
print("Scene data path: {}".format(scene_data_path))

# load gt mesh if exists
gt_meshes_paths = [os.path.join("debug/meshes/", dataset_name, scene_name, "mesh.ply")]

# dataset loading
dataset_train = MVDataset(
    dataset_name,
    scene_name,
    scene_data_path,
    point_clouds_paths=gt_meshes_paths,
    split="train",
    use_every_for_test_split=8,
    auto_center_method="none",  # "poses", "focus", "none"
    auto_orient_method="none",  # "up", "none"
    # auto_scale_poses=False,
    profiler=profiler,
    device=device,
)

# Visualize cameras
# fig = plot_cameras(
#     dataset_train.cameras,
#     points=dataset_train.point_clouds[0],
#     azimuth_deg=20,
#     elevation_deg=30,
#     up="y",
#     figsize=(15, 15),
# )

# plt.show()
# plt.savefig("test_static_scenes.png", bbox_inches="tight", pad_inches=0)

# TODO: plot_projected_points
# plt.savefig("test_static_scenes_projection.png", transparent=True)

# Visualize camera rays
# camera = dataset_train.cameras[0]
# fig = plot_camera_rays(
#     camera, 512, azimuth_deg=60, elevation_deg=30, up="y", figsize=(15, 15)
# )

# plt.show()

from datasets.utils.loader import DatasetSampler, custom_collate, get_next_batch

# Create a DataLoader for the MVDataset
cameras_batch_size = len(dataset_train.cameras)  # alway sample from all cameras
max_rays_batch_size = 512
per_camera_rays_batch_size = max_rays_batch_size // cameras_batch_size
dataset_train.per_camera_rays_batch_size = per_camera_rays_batch_size
# nr_workers = 1
custom_sampler = DatasetSampler(dataset_train, shuffle=True)
data_loader = DataLoader(
    dataset=dataset_train,
    batch_size=cameras_batch_size,
    sampler=custom_sampler,
    collate_fn=custom_collate,
    # num_workers=nr_workers,
)

nr_iterations = 100
for i in tqdm(range(nr_iterations)):
    if profiler is not None:
        profiler.start("get_next_batch")

    # get rays and gt values
    with torch.set_grad_enabled(False):
        idx, rays_o, rays_d, gt_rgb, gt_mask, frame_idx = get_next_batch(data_loader)

    if profiler is not None:
        profiler.end("get_next_batch")

# for i in tqdm(range(nr_iterations)):
#     profiler.start("get_next_batch")
#     idx, rays_o, rays_d, rgb, mask = get_next_batch(data_loader)
#     profiler.end("get_next_batch")

#     fig = plot_current_batch(
#         dataset_train.cameras,
#         idx,
#         rays_o,
#         rays_d,
#         rgb,
#         mask,
#         azimuth_deg=60,
#         elevation_deg=30,
#         up="y",
#         figsize=(15, 15),
#     )

#     # plt.show()
#     plt.savefig(f"test_static_scenes_batch_{i}.png", bbox_inches="tight", pad_inches=0)

if profiler is not None:
    profiler.print_avg_times()
