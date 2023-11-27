import os
import sys
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
dataset_name = "dtu"
scene_name = "dtu_scan83"

# load gt mesh if exists
gt_meshes_paths = [os.path.join("debug/meshes/", dataset_name, scene_name, "mesh.ply")]

# dataset loading
mv_data = MVDataset(
    dataset_name,
    scene_name,
    datasets_path,
    point_clouds_paths=gt_meshes_paths,
    splits=["train", "test"],
    test_camera_freq=8,
    load_mask=True,
)

camera = deepcopy(mv_data["test"][0])
print(camera)

# make from the gt frame a smaller frame until we reach a certain size
while min(camera.width, camera.height) > 100:
    print("camera.width", camera.width, "camera.height", camera.height)
    camera.subsample(scale=0.5)
print("camera.width", camera.width, "camera.height", camera.height)

# gen rays
rays_o, rays_d, points_2d = get_camera_rays(camera)
print("rays_o", rays_o.shape, rays_o.device)
print("rays_d", rays_d.shape, rays_d.device)
print("points_2d", points_2d.shape, points_2d.device)

rgb, mask, _ = get_camera_frames(camera, points_2d=points_2d)
print("rgb", rgb.shape, rgb.device)
print("mask", mask.shape, mask.device)

# visualize camera
fig = plot_camera_rays(
    camera, 512, azimuth_deg=60, elevation_deg=30, up="y", figsize=(15, 15)
)

plt.show()
# plt.savefig(os.path.join("imgs", "dtu_camera_rays.png"), bbox_inches="tight", pad_inches=0, dpi=300)
# plt.close()