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

from mvdatasets.utils.plotting import plot_bounding_boxes
from mvdatasets.utils.profiler import Profiler
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.common import get_dataset_test_preset
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.virtual_cameras import sample_cameras_on_hemisphere
from mvdatasets.utils.bounding_box import BoundingBox
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
        device = "cuda"
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
    bounding_boxes = []
    
    bb = BoundingBox(
        pose=np.eye(4),
        local_scale=np.array([1.0, 1.0, 1.0]),
        device=device
    )
    bounding_boxes.append(bb)
    
    points = bb.get_random_points_inside(1000)
    # get points norm
    print("min", torch.min(points, dim=0)[0])
    print("max", torch.max(points, dim=0)[0])
    print("points", points)
    
    # bb_pose = np.eye(4)
    # bb_pose[:3, 3] = np.array([0.0, 0.0, 0.0])
    # bb = BoundingBox(
    #     pose=bb_pose,
    #     sizes=np.array([0.6, 0.4, 0.2]),
    # )
    # bounding_boxes.append(bb)
    
    # # test save
    # bb.save_as_ply(".", "bb_0")
    
    # bb_pose = np.eye(4)
    # bb_pose[:3, 3] = np.array([1.0, 0.0, 0.0])
    # bb_scale = np.array([0.7, 0.8, 0.9])
    # bb_pose[:3, :3] = rot_y_3d(deg2rad(45)) @ rot_x_3d(deg2rad(45))
    # bb_pose[:3, :3] *= bb_scale
    # bb = BoundingBox(
    #     pose=bb_pose,
    #     father_bb=bounding_boxes[0],
    # )
    # bounding_boxes.append(bb)
    
    # bb_pose = np.eye(4)
    # bb_pose[:3, 3] = np.array([-0.5, 0.5, 0.0])
    # bb_scale = np.array([0.4, 0.3, 0.2])
    # bb_pose[:3, :3] = rot_y_3d(deg2rad(45)) @ rot_x_3d(deg2rad(45))
    # bb_pose[:3, :3] *= bb_scale
    # bb = BoundingBox(
    #     pose=bb_pose,
    #     father_bb=bounding_boxes[0],
    # )
    # bounding_boxes.append(bb)
    
    # bb_pose = np.eye(4)
    # bb_center = np.array([0.2, 0.5, -0.5])
    # bb_pose[:3, 3] = bb_center
    # bb_pose[:3, :3] = rot_x_3d(deg2rad(30))
    # bb = BoundingBox(
    #     pose=bb_pose,
    #     sizes=np.array([0.5, 0.4, 0.3]),
    # )
    # bounding_boxes.append(bb)
    
    # bb_pose = np.eye(4)
    # bb_center = np.array([-0.2, -0.5, 0.5])
    # bb_pose[:3, 3] = bb_center
    # bb_pose[:3, :3] = rot_y_3d(deg2rad(30))
    # bb = BoundingBox(
    #     father_bb=bounding_boxes[0],  # parent bounding box
    #     pose=bb_pose,
    #     sizes=np.array([0.3, 0.4, 0.2]),
    # )
    # bounding_boxes.append(bb)
    
    # bb_pose = np.eye(4)
    # bb_center = np.array([0.2, 0.1, 0.6])
    # bb_pose[:3, 3] = bb_center
    # bb_pose[:3, :3] = rot_z_3d(deg2rad(30))
    # bb = BoundingBox(
    #     father_bb=bounding_boxes[0],  # parent bounding box
    #     pose=bb_pose,
    #     sizes=np.array([0.5, 0.1, 0.4]),
    # )
    # bounding_boxes.append(bb)
    
    # shoot rays from camera and intersect with boxes
    rays_o, rays_d, points_2d = get_camera_rays(camera, device="cuda")
    
    intersections = []
    for i, bb in enumerate(bounding_boxes):
        is_hit, t_near, t_far, p_near, p_far = bb.intersect(rays_o, rays_d)
        intersections.append([is_hit, t_near, t_far, p_near, p_far])
        print(f"is {i} bb hit?", np.any(is_hit.cpu().numpy()))
    
    points_3d = []
    for (is_hit, _, _, p_near, p_far) in intersections:
        points_3d.append(p_near[is_hit].cpu().numpy())
        points_3d.append(p_far[is_hit].cpu().numpy())
    points_3d = np.concatenate(points_3d, axis=0)
    
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
    
    plt.imshow(near_depth, cmap="jet")
    plt.colorbar()
    plt.savefig(
        os.path.join("plots", "ray_bb_hit_near_depth.png"),
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()
    
    plt.imshow(far_depth, cmap="jet")
    plt.colorbar()
    plt.savefig(
        os.path.join("plots", "ray_bb_hit_far_depth.png"),
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()
    
    # # visualize
    # fig = plot_cameras(
    #     camera,
    #     nr_rays=32,
    #     points_3d=points_3d,
    #     bounding_boxes=bounding_boxes,
    #     azimuth_deg=20,
    #     elevation_deg=30,
    #     radius=camera_radius,
    #     up="z",
    #     draw_bounding_cube=False,
    #     figsize=(15, 15),
    #     title="camera, rays and bounding boxes intersection",
    # )
    # # plt.show()
    
    # plt.savefig(
    #     os.path.join("plots", "camera_rays_bbs_intersections.png"),
    #     transparent=False,
    #     bbox_inches="tight",
    #     pad_inches=0,
    #     dpi=300
    # )
    # plt.close()
    
    # fig = plot_bounding_boxes(
    #     bounding_boxes=bounding_boxes,
    #     azimuth_deg=230,
    #     elevation_deg=60,
    #     radius=1.0,
    #     up="z",
    #     figsize=(15, 15),
    #     draw_origin=False,
    #     draw_frame=False,
    #     title="",
    # )
    # # plt.show()
    
    # plt.savefig(
    #     os.path.join("plots", f"bbs.png"),
    #     transparent=False,
    #     bbox_inches="tight",
    #     pad_inches=0,
    #     dpi=300
    # )
    # plt.close()