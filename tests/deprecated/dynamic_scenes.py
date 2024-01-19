# TODO: deprecated, needs to be updated

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
from mvdatasets.utils.plotting import plot_cameras
from mvdatasets.utils.geometry import (
    apply_transformation_3d,
    rot_x_3d,
    rot_y_3d,
    rot_z_3d,
    project_points_3d_to_2d,
)
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.images import tensor2image

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
print("scene data path: {}".format(scene_data_path))

# if dataset_name == "fluid_sym":
#     # load gt particles
#     original_data_path = os.path.join("/home/stefano/Data/fluid_sym_data/", scene["name"])
#     particles_path = os.path.join(original_data_path, "frames", "particles-000000000.dat")
#     points_3d = load_particles_fluidsym(particles_path)

# load gt particles
particles_dir = os.path.join("debug", "particles", dataset_name, scene_name)
particles_files = os.listdir(particles_dir)
# sort by increasing int
particles_files = sorted(particles_files, key=lambda x: int(x.split(".")[0]))
particles_paths = []
for file in particles_files:
    particles_paths.append(os.path.join(particles_dir, file))

# dataset loading
training_data = MVDataset(
    dataset_name,
    scene_name,
    scene_data_path,
    point_clouds_paths=particles_paths,
    split="all",
    load_mask=True,
    auto_center_method="none",  # "poses", "focus", "none"
    auto_orient_method="none",  # "up", "none"
    device=device,
)

# shift = np.array([0.5, 0.0, 0.5])
# points_3d += shift

fig = plot_cameras(
    training_data.cameras,
    points_3d=training_data.point_clouds[0],
    azimuth_deg=20,
    elevation_deg=30,
    up="y",
    figsize=(15, 15),
)

# plt.show()
plt.savefig("test_dynamic_scenes.png", bbox_inches="tight", pad_inches=0, dpi=300)

img_torch = training_data.cameras[0].imgs[0]
print("img_torch", img_torch.shape)
img_pil = tensor2image(img_torch)
img_pil.save("test_dynamic_scenes_img.png")

img_torch = training_data.cameras[0].masks[0]
print("img_torch", img_torch.shape)
img_pil = tensor2image(img_torch)
img_pil.save("test_dynamic_scenes_mask.png")

camera_idx = 1
frame_idx_idx = 7
img_np = training_data.cameras[camera_idx].get_rgb(frame_idx_idx).cpu().numpy()
intrinsics = training_data.cameras[camera_idx].intrinsics.cpu().numpy()
points_2d = project_points_3d_to_2d(
    points_3d=training_data.point_clouds[frame_idx_idx],
    intrinsics=intrinsics,
    c2w=training_data.cameras[camera_idx].get_pose().cpu().numpy(),
)
# filter out points outside image range
points_2d = points_2d[points_2d[:, 0] > 0]
points_2d = points_2d[points_2d[:, 1] > 0]
points_2d = points_2d[points_2d[:, 0] < img_np.shape[1]]
points_2d = points_2d[points_2d[:, 1] < img_np.shape[0]]
print("points_2d", points_2d.shape)

fig = plt.figure()
plt.imshow(img_np, alpha=1.0)
colors = np.column_stack([points_2d, np.zeros((points_2d.shape[0], 1))])
colors /= np.max(colors)
colors += 0.5
colors /= np.max(colors)
plt.scatter(points_2d[:, 0], points_2d[:, 1], s=5, c=colors, marker=".")
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("test_dynamic_scenes_projection.png", dpi=300)
