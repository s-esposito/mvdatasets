import numpy
import PIL
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import struct

from datasets.loaders.pac_nerf import load_particles_pacnerf
from datasets.visualization.matplotlib import plot_cameras
from datasets.utils.geometry import transform_points_3d, rot_x_3d, rot_y_3d, rot_z_3d, project_points_3d_to_2d
from datasets.mv_dataset import MVDataset

torch.manual_seed(42)
torch.set_default_dtype(torch.float32)
device = "cpu"  # "cuda:0"
torch.set_default_device(device)

data_path = "/home/stefano/Data"
dataset_name = "pac_nerf"
scene_name = "torus"

scene_data_path = os.path.join(data_path, dataset_name, scene_name)
# make sure folder exists
assert os.path.exists(scene_data_path), "Scene data path does not exist"
print("Scene data path: {}".format(scene_data_path))

# dataset loading
training_data = MVDataset(
    dataset_name,
    scene_name,
    scene_data_path,
    split="train",
    load_with_mask=False,
    auto_center_method="none",  # "poses", "focus", "none"
    auto_orient_method="none",  # "up", "none"
    device=device,
)

# if dataset_name == "fluid_sym":
#     # load gt particles
#     original_data_path = os.path.join("/home/stefano/Data/fluid_sym_data/", scene["name"])
#     particles_path = os.path.join(original_data_path, "frames", "particles-000000000.dat")
#     points_3d = load_particles_fluidsym(particles_path)

# elif dataset_name == "pac_nerf":
# load gt particles
particles_path = os.path.join("debug", "particles", dataset_name, scene_name, "0.ply")
# make sure folder exists
if not os.path.exists(scene_data_path):
    print(f"scene point cloud not found in {scene_data_path}")
    points_3d = np.zeros((0, 3))
else:
    points_3d = load_particles_pacnerf(particles_path)

# shift = np.array([0.5, 0.0, 0.5])
# points_3d += shift

fig = plot_cameras(training_data.cameras, points=points_3d, azimuth_deg=15, elevation_deg=15, up="y", figsize=(15, 15))
plt.show()

# save fig to file
# fig.savefig(os.path.join("plots", f"{scene['dataset']}-{scene['name']}.png"), bbox_inches='tight', pad_inches=0)

# # project points to 2d

# if points_3d.shape[0] > 0:
#     camera_idx = 5
#     points_2d = project_points_3d_to_2d(points=points_3d, camera=training_data.cameras[camera_idx])
#     fig = plt.figure()
#     # flip image vertically for visualization
#     plt.imshow(np.fliplr(training_data.cameras[camera_idx].img), origin="lower")
#     # plt.imshow(cameras[0].img)
#     # visualize 2d points
#     plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, c="r")
#     # axis equal
#     plt.gca().set_aspect("equal", adjustable="box")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.show()
