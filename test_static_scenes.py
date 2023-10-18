import numpy
import PIL
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from datasets.visualization.matplotlib import plot_cameras
from datasets.mv_dataset import MVDataset

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

# dataset loading
training_data = MVDataset(
    dataset_name,
    scene_name,
    scene_data_path,
    split="train",
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

# load gt mesh if exists
gt_mesh_path = os.path.join("debug/meshes/", dataset_name, scene_name, "mesh.ply")
# if exists, load it
if os.path.exists(gt_mesh_path):
    print("Found gt mesh at {}".format(gt_mesh_path))
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    points_3d = np.array(gt_mesh.vertices)
    if points_3d.shape[0] > 1000:
        # downsample
        random_idx = np.random.choice(points_3d.shape[0], 10000, replace=False)
        points_3d = points_3d[random_idx]
    print("Loaded {} points from mesh".format(points_3d.shape[0]))
else:
    print("No gt mesh found at {}".format(gt_mesh_path))
    points_3d = np.empty((0, 3))

fig = plot_cameras(training_data.cameras, points=points_3d, azimuth_deg=20, elevation_deg=30, up="y")

plt.show()
