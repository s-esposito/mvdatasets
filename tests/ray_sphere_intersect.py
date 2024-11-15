import numpy as np
import PIL
import os
import sys
import time
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import imageio

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.utils.profiler import Profiler
from mvdatasets.mvdataset import MVDataset
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.utils.virtual_cameras import sample_cameras_on_hemisphere
from mvdatasets.utils.bounding_sphere import BoundingSphere
from mvdatasets.utils.geometry import deg2rad, rot_x_3d, rot_y_3d, rot_z_3d
from mvdatasets.utils.raycasting import get_camera_rays


if __name__ == "__main__":

    # Set a random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # # Check if CUDA (GPU support) is available
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
    
    width = 800
    height = 800
    vfov = 90.0
    focal = (height / 2) / np.tan(np.deg2rad(vfov / 2))
    cx = width / 2
    cy = height / 2
    intrinsics = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    camera_radius = 0.5
    
    sampled_cameras = sample_cameras_on_hemisphere(
        intrinsics=intrinsics,
        width=width,
        height=height,
        radius=camera_radius,
        nr_cameras=1
    )
    camera = sampled_cameras[0]

    # create bounding boxes
    bounding_spheres = []

    sphere = BoundingSphere(
        pose=np.eye(4),
        local_scale=0.4,
        device=device
    )
    bounding_spheres.append(sphere)
    
    # shoot rays from camera and intersect with boxes
    rays_o, rays_d, points_2d = get_camera_rays(camera, device=device)
    
    intersections = []
    for i, bb in enumerate(bounding_spheres):
        is_hit, t_near, t_far, p_near, p_far = bb.intersect(rays_o, rays_d)
        intersections.append([is_hit, t_near, t_far, p_near, p_far])
        print(f"is sphere {i} hit?", np.any(is_hit.cpu().numpy()))
    
    near_depth = np.ones((height, width)).flatten() * np.inf
    far_depth = np.ones((height, width)).flatten() * np.inf
    for (is_hit, t_near, t_far, _, _) in intersections:
        # t_near
        t_near_np = t_near.cpu().numpy()
        updates = t_near_np < near_depth
        updates *= is_hit.cpu().numpy()
        near_depth[updates] = t_near_np[updates]
        # t_far
        t_far_np = t_far.cpu().numpy()
        updates = t_far_np < far_depth
        updates *= is_hit.cpu().numpy()
        far_depth[updates] = t_far_np[updates]
    
    near_depth = near_depth.reshape(height, width)
    far_depth = far_depth.reshape(height, width)
    
    max_depth = np.max(far_depth)
    min_depth = np.min(near_depth)
    plt.imshow(near_depth, cmap="jet")  # , vmin=0.0)  # , vmax=max_depth)
    plt.colorbar()
    plt.savefig(
        os.path.join("plots", "ray_sphere_hit_near_depth.png"),
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()
    
    plt.imshow(far_depth, cmap="jet")  # , vmin=0.0)  # , vmax=max_depth)
    plt.colorbar()
    plt.savefig(
        os.path.join("plots", "ray_sphere_hit_far_depth.png"),
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()