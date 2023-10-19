import numpy
import PIL
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import struct

# from datasets.loaders.pac_nerf import load_particles_pacnerf
from datasets.utils.plotting import plot_cameras
from datasets.utils.geometry import (
    transform_points_3d,
    rot_x_3d,
    rot_y_3d,
    rot_z_3d,
    project_points_3d_to_2d,
)
from datasets.mv_dataset import MVDataset
from datasets.utils.images import tensor2image

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

# if dataset_name == "fluid_sym":
#     # load gt particles
#     original_data_path = os.path.join("/home/stefano/Data/fluid_sym_data/", scene["name"])
#     particles_path = os.path.join(original_data_path, "frames", "particles-000000000.dat")
#     points_3d = load_particles_fluidsym(particles_path)

# elif dataset_name == "pac_nerf":
# load gt particles
particles_path = os.path.join("debug", "particles", dataset_name, scene_name, "0.ply")
# # make sure folder exists
# if not os.path.exists(scene_data_path):
#     print(f"scene point cloud not found in {scene_data_path}")
#     points_3d = np.zeros((0, 3))
# else:
#     points_3d = load_particles_pacnerf(particles_path)

# dataset loading
training_data = MVDataset(
    dataset_name,
    scene_name,
    scene_data_path,
    point_cloud_path=particles_path,
    split="train",
    load_with_mask=True,
    auto_center_method="none",  # "poses", "focus", "none"
    auto_orient_method="none",  # "up", "none"
    device=device,
)

# shift = np.array([0.5, 0.0, 0.5])
# points_3d += shift

fig = plot_cameras(
    training_data.cameras,
    points=training_data.point_cloud,
    azimuth_deg=20,
    elevation_deg=30,
    up="z",
    figsize=(15, 15),
)

# plt.show()
plt.savefig("test_dynamic_scenes.png", bbox_inches="tight", pad_inches=0)

img_torch = training_data.cameras[0].imgs[0]
print("img_torch", img_torch.shape)
img_pil = tensor2image(img_torch)
img_pil.save("test_dynamic_scenes_img.png")

img_torch = training_data.cameras[0].masks[0]
print("img_torch", img_torch.shape)
img_pil = tensor2image(img_torch)
img_pil.save("test_dynamic_scenes_mask.png")

camera_idx = 0
w2c = np.linalg.inv(training_data.cameras[camera_idx].get_pose().cpu().numpy())
intrinsics = training_data.cameras[camera_idx].intrinsics.cpu().numpy()
points_2d = project_points_3d_to_2d(
    points=training_data.point_cloud, intrinsics=intrinsics, w2c=w2c
)
img_np = training_data.cameras[camera_idx].get_frame().cpu().numpy()
print("points_2d", points_2d.shape)

fig = plt.figure()
plt.imshow(img_np)
plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, c="r")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("test_dynamic_scenes_projection.png")
