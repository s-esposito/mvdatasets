import numpy
import PIL
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from datasets.utils.plotting import plot_cameras
from datasets.mv_dataset import MVDataset
from datasets.utils.images import tensor2image
from datasets.utils.geometry import project_points_3d_to_2d

torch.manual_seed(42)
torch.set_default_dtype(torch.float32)
device = "cuda:0"
torch.set_default_device(device)

data_path = "/home/stefano/Data"
dataset_name = "dtu"
scene_name = "dtu_scan83"

scene_data_path = os.path.join(data_path, dataset_name, scene_name)
# make sure folder exists
assert os.path.exists(scene_data_path), "Scene data path does not exist"
print("Scene data path: {}".format(scene_data_path))

# load gt mesh if exists
gt_mesh_path = os.path.join("debug/meshes/", dataset_name, scene_name, "mesh.ply")
# # if exists, load it
# if os.path.exists(gt_mesh_path):
#     print("Found gt mesh at {}".format(gt_mesh_path))
#     gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
#     points_3d = np.array(gt_mesh.vertices)
#     if points_3d.shape[0] > 1000:
#         # downsample
#         random_idx = np.random.choice(points_3d.shape[0], 10000, replace=False)
#         points_3d = points_3d[random_idx]
#     print("Loaded {} points from mesh".format(points_3d.shape[0]))
# else:
#     print("No gt mesh found at {}".format(gt_mesh_path))
#     points_3d = np.empty((0, 3))

# dataset loading
training_data = MVDataset(
    dataset_name,
    scene_name,
    scene_data_path,
    point_cloud_path=gt_mesh_path,
    split="all",
    auto_center_method="none",  # "poses", "focus", "none"
    auto_orient_method="none",  # "up", "none"
    # auto_scale_poses=False,
    device=device,
)

# T = training_data.cameras[0].transform
# K = training_data.cameras[0].intrinsics
# Rt = training_data.cameras[0].get_pose()
# P = training_data.cameras[0].get_projection()

# print("T", T)
# print("K", K)
# print("Rt", Rt)
# print("P", P)

fig = plot_cameras(
    training_data.cameras,
    points=training_data.point_cloud,
    azimuth_deg=20,
    elevation_deg=30,
    up="z",
    figsize=(15, 15),
)

# plt.show()
plt.savefig("test_static_scenes.png", bbox_inches="tight", pad_inches=0)

img_torch = training_data.cameras[0].imgs[0]
print("img_torch", img_torch.shape)
img_pil = tensor2image(img_torch)
img_pil.save("test_static_scenes_img.png")

img_torch = training_data.cameras[0].masks[0]
print("img_torch", img_torch.shape)
img_pil = tensor2image(img_torch)
img_pil.save("test_static_scenes_mask.png")

camera_idx = 0
img_np = training_data.cameras[camera_idx].get_frame().cpu().numpy()
w2c = np.linalg.inv(training_data.cameras[camera_idx].get_pose().cpu().numpy())
intrinsics = training_data.cameras[camera_idx].intrinsics.cpu().numpy()
points_2d = project_points_3d_to_2d(
    points=training_data.point_cloud, intrinsics=intrinsics, w2c=w2c
)
# filter out points outside image range
points_2d = points_2d[points_2d[:, 0] > 0]
points_2d = points_2d[points_2d[:, 1] > 0]
points_2d = points_2d[points_2d[:, 1] < img_np.shape[1]]
points_2d = points_2d[points_2d[:, 0] < img_np.shape[0]]
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
plt.savefig("test_static_scenes_projection.png", transparent=True)
