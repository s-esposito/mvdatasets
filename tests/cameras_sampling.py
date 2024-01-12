import numpy
import PIL
import os
import sys
import time
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import imageio

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.utils.plotting import plot_cameras
from mvdatasets.utils.plotting import plot_current_batch
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.common import get_dataset_test_preset
from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.tensor_reel import TensorReel

import numpy as np
# from scipy.spatial.transform import Rotation as R


def look_at(eye, center, up):
    """Compute camera pose from look at vectors
    args:
        eye (np.ndarray) : (3,) camera position
        center (np.ndarray) : (3,) point to look at
        up (np.ndarray) : (3,) up vector
    out:
        pose (np.ndarray) : (4, 4) camera pose
    """
    
    assert eye.shape == (3,)
    assert center.shape == (3,)
    assert up.shape == (3,)
    
    # get camera frame
    z = eye - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    
    # get rotation matrix
    rotation = np.eye(3)
    rotation[:3, 0] = x
    rotation[:3, 1] = y
    rotation[:3, 2] = z
    
    return rotation


def sample_cameras_on_hemisphere(intrinsics, width, height, radius=1, nr_cameras=10):
    
    # # azimuth_deg = np.linspace(0, 360, nr_cameras, endpoint=False)
    # # elevation_deg = np.linspace(0, 45, nr_cameras, endpoint=False)
    # azimuth_deg = np.random.uniform(0, 360, nr_cameras)
    # elevation_deg = np.random.uniform(-90, 90, nr_cameras)
    # azimuth_rad = deg2rad(azimuth_deg)
    # elevation_rad = deg2rad(elevation_deg)
    # x = np.cos(azimuth_rad) * np.cos(elevation_rad) * radius
    # y = np.sin(azimuth_rad) * np.cos(elevation_rad) * radius
    # z = np.sin(elevation_rad) * radius
    # x = np.array(x)
    # y = np.array(y)
    # z = np.array(z)
    # cameras_centers = np.column_stack((x, y, z))
    
    points = np.random.uniform(-1, 1, (nr_cameras, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    cameras_centers = points * radius
    
    up = np.array([0, 1, 0])
    center = np.array([0, 0, 0])
    
    cameras = []
    for i in range(nr_cameras):

        # get rotation matrix from azimuth and elevation
        pose = np.eye(4)
        rotation = look_at(cameras_centers[i], center, up)
        pose[:3, :3] = rotation
        pose[:3, 3] = cameras_centers[i]
        
        # local transform
        local_transform = np.eye(4)
        local_transform[:3, :3] = np.array([[-1, 0, 0],
                                            [0, -1, 0],
                                            [0, 0, -1]], dtype=np.float32)
    
        camera = Camera(
                            intrinsics,
                            pose,
                            width=width,
                            height=height,
                            local_transform=local_transform,
                            camera_idx=i
                        )
        cameras.append(camera)
    
    return cameras


if __name__ == "__main__":

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

    # Get dataset test preset
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "dtu"
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True
    )

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        point_cloud = np.array([[0, 0, 0]])
    
    intrinsics = mv_data["train"][0].get_intrinsics()
    width = mv_data["train"][0].width
    height = mv_data["train"][0].height
    camera_center = mv_data["train"][0].get_center()
    camera_radius = np.linalg.norm(camera_center)
    
    sampled_cameras = sample_cameras_on_hemisphere(
        intrinsics=intrinsics,
        width=width,
        height=height,
        radius=camera_radius,
        nr_cameras=100
    )

    # Visualize cameras
    fig = plot_cameras(
        sampled_cameras,
        points=point_cloud,
        azimuth_deg=20,
        elevation_deg=30,
        up="y",
        figsize=(15, 15),
        title="sampled cameras",
    )

    # plt.show()
    plt.savefig(
        os.path.join("imgs", f"{dataset_name}_sampled_cameras.png"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()
    
    # Create tensor reel
    tensor_reel = TensorReel(sampled_cameras, width=width, height=height, device=device)
    
    benchmark = False
    batch_size = 512
    nr_iterations = 10
    cameras_idx = None
    pbar = tqdm(range(nr_iterations), desc="ray casting", ncols=100)
    azimuth_deg = 0
    azimuth_deg_delta = 360 / (nr_iterations / 2)
    frames_paths = []
    for i in pbar:

        # cameras_idx = np.random.permutation(len(mv_data["train"]))[:2]

        if profiler is not None:
            profiler.start("get_next_batch")

        # get rays and gt values
        (
            camera_idx,
            rays_o,
            rays_d,
            _,
            frame_idx,
        ) = tensor_reel.get_next_batch(
            batch_size=batch_size, cameras_idx=cameras_idx,
        )

        if profiler is not None:
            profiler.end("get_next_batch")

        if not benchmark:
        
            print("camera_idx", camera_idx.shape, camera_idx.device)
            print("rays_o", rays_o.shape, rays_o.device)
            print("rays_d", rays_d.shape, rays_d.device)
            
            fig = plot_current_batch(
                mv_data["train"],
                camera_idx,
                rays_o,
                rays_d,
                azimuth_deg=azimuth_deg,
                elevation_deg=30,
                up="y",
                figsize=(15, 15),
            )

            # plt.show()
            frame_path = os.path.join("plots", f"{dataset_name}_sampled_cameras_batch_{i}.png")
            plt.savefig(
                frame_path,
                bbox_inches="tight",
                pad_inches=0,
                dpi=72,
                transparent=True
            )
            plt.close()
            frames_paths.append(frame_path)

            # update azimuth every 2 iterations
            if i % 2 != 0:
                azimuth_deg += azimuth_deg_delta
            
    if profiler is not None:
        profiler.print_avg_times()