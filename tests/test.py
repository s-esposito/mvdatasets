import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.utils.plotting import plot_points_2d_on_image
from mvdatasets.mvdataset import MVDataset
from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.geometry import linear_transformation_3d, perspective_projection, inv_perspective_projection

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

imgs = np.zeros((1, 1360, 1360, 3))
intrinsics = np.array(
    [
        [785.0, 0.0, 680.0],
        [0.0, 785.0, 680.0],
        [0.0, 0.0, 1.0],
    ]
)
pose = np.array([
    [0.38, -0.82, 0.42, 1.3],
    [0.87, 0.17, -0.45, 2.0],
    [0.29, 0.54, 0.78, -1.5],
    [0.0, 0.0, 0.0, 1.0]
])

camera = Camera(imgs=imgs, pose=pose, intrinsics=intrinsics)
print("camera", camera)

points_3d = np.array([[3, 2, 6]])
print("points_3d", points_3d)

camera_center = camera.get_pose()[:3, 3]
points_depth = np.linalg.norm(points_3d - camera_center, axis=1)
print("points_depth", points_depth)

# augmented_points_3d = augment_vectors(points_3d)
# print("augmented_points_3d", augmented_points_3d)

# K = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
# print("K", K)

# Rt = pose
# print("Rt", Rt)

# P = K @ Rt
# print("P", P)

# homogeneous_points_2d = (P @ augmented_points_3d.T).T
# print("homogeneous_points_2d", homogeneous_points_2d)
# augmented_points_2d = homogeneous_points_2d / homogeneous_points_2d[:, 2:]
# print("augmented_points_2d", augmented_points_2d)

# points_2d = augmented_points_2d[:, :2]
# print("points_2d", points_2d)

# Rt = pose
# print("Rt", Rt)

# homogeneous_points_3d = (Rt @ augmented_points_3d.T).T
# print("homogeneous_points_3d", homogeneous_points_3d)
# augmented_points_3d = homogeneous_points_3d / homogeneous_points_3d[:, 3:]
# print("augmented_points_3d", augmented_points_3d)
# points_3d = augmented_points_3d[:, :3]
# print("points_3d", points_3d)

# points_2d = perspective_projection(intrinsics, points_3d)
# print("points_2d", points_2d)

Rt = pose
points_3d = linear_transformation_3d(points_3d, Rt)
print("points_3d", points_3d)

points_depth = np.linalg.norm(points_3d, axis=1)
print("points_depth", points_depth)

K = intrinsics
points_2d = perspective_projection(K, points_3d)
print("points_2d", points_2d)
points_3d = inv_perspective_projection(np.linalg.inv(K), points_2d)
print("points_3d", points_3d)