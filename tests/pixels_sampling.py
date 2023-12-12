import sys
import os
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.utils.plotting import plot_points_2d_on_image
from mvdatasets.utils.raycasting import get_camera_rays
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

# Set datasets path
datasets_path = "/home/stefano/Data"

# # test DTU
# dataset_name = "dtu"
# scene_name = "dtu_scan83"
# pc_path = "debug/meshes/dtu/dtu_scan83.ply"

# # test blender
# dataset_name = "blender"
# scene_name = "lego"
# pc_path = "debug/point_clouds/blender/lego.ply"

# test blendernerf
dataset_name = "blendernerf"
scene_name = "plushy"
pc_path = "debug/meshes/blendernerf/plushy.ply"

# # test all
# dataset_names = ["dtu", "blender", "blendernerf"]
# scene_names = ["dtu_scan83", "lego"]
# pc_paths = ["debug/meshes/dtu/dtu_scan83.ply", "debug/point_clouds/blender/lego.ply", "debug/meshes/blendernerf/plushy.ply"]

# for dataset_name, scene_name, pc_path in zip(dataset_names, scene_names, pc_paths):

# dataset loading
mv_data = MVDataset(
    dataset_name,
    scene_name,
    datasets_path,
    point_clouds_paths=[pc_path],
    splits=["train", "test"]
)

# random camera index
rand_idx = torch.randint(0, len(mv_data["test"]), (1,)).item()
camera = deepcopy(mv_data["test"][rand_idx])
print(camera)

# resize camera's rgb modality
camera.resize(max_dim=100)

# gen rays
rays_o, rays_d, points_2d = get_camera_rays(camera, jitter_pixels=True)
fig = plot_points_2d_on_image(
    camera,
    points_2d[:, [1, 0]],
    show_ticks=True,
    figsize=(15, 15)
)
plt.savefig(
    os.path.join("imgs", f"{dataset_name}_screen_space_sampling_jittered.png"),
    transparent=True,
    dpi=300
)
plt.close()

# gen rays
rays_o, rays_d, points_2d = get_camera_rays(camera, jitter_pixels=False)

fig = plot_points_2d_on_image(
    camera,
    points_2d[:, [1, 0]],
    show_ticks=True,
    figsize=(15, 15)
)
plt.savefig(
    os.path.join("imgs", f"{dataset_name}_screen_space_sampling.png"),
    transparent=True,
    dpi=300
)
plt.close()