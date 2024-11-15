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

from mvdatasets.utils.plotting import plot_cameras
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.utils.bounding_box import BoundingBox
from mvdatasets.utils.bounding_sphere import BoundingSphere
from mvdatasets.utils.printing import print_error, print_warning
from mvdatasets.config import datasets_path


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
    
    print("scene_radius:", mv_data.scene_radius)
    print("init_sphere_radius:", mv_data.init_sphere_radius)
    
    if mv_data.init_sphere_radius > mv_data.scene_radius:
        print_error("init_sphere_radius > scene_radius, this can't be true")

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        point_cloud = np.array([[0, 0, 0]])
        
    # create bounding primitives
    bounding_boxes = []
    bounding_spheres = []
    
    # scene
    if mv_data.scene_type == "bounded":
        bb = BoundingBox(
            pose=np.eye(4),
            local_scale=mv_data.scene_radius*2,
            line_width=2.0,
            device=device
        )
        bounding_boxes.append(bb)
    elif mv_data.scene_type == "unbounded":
        bs = BoundingSphere(
            pose=np.eye(4),
            local_scale=0.5,  # outer primitive
            device=device,
            verbose=True
        )
        bounding_spheres.append(bs)
    
    # sdf init
    
    bs = BoundingSphere(
        pose=np.eye(4),
        local_scale=np.array([mv_data.init_sphere_radius, mv_data.init_sphere_radius, mv_data.init_sphere_radius]),  # outer primitive
        device=device,
        verbose=True
    )
    bounding_spheres.append(bs)
    
    draw_contraction_spheres = False
    scene_type = config.get("scene_type", None)
    if scene_type == "unbounded":
        draw_contraction_spheres = True
    print("draw_contraction_spheres:", draw_contraction_spheres)

    # Visualize cameras
    fig = plot_cameras(
        mv_data["train"],
        points_3d=point_cloud,
        bounding_boxes=bounding_boxes,
        bounding_spheres=bounding_spheres,
        azimuth_deg=20,
        elevation_deg=30,
        up="z",
        scene_radius=1.0,  # mv_data.max_camera_distance,
        draw_bounding_cube=True,
        draw_image_planes=True,
        draw_cameras_frustums=False,
        draw_contraction_spheres=draw_contraction_spheres,
        figsize=(15, 15),
        title="training cameras",
    )

    # plt.show()
    plt.savefig(
        os.path.join("plots", f"{dataset_name}_training_cameras.png"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()

    # Visualize cameras
    fig = plot_cameras(
        mv_data["test"],
        points_3d=point_cloud,
        points_3d_color="black",
        bounding_boxes=bounding_boxes,
        bounding_spheres=bounding_spheres,
        azimuth_deg=20,
        elevation_deg=30,
        up="z",
        scene_radius=1.0,  # mv_data.max_camera_distance,
        draw_bounding_cube=True,
        draw_image_planes=True,
        draw_cameras_frustums=False,
        draw_contraction_spheres=draw_contraction_spheres,
        figsize=(15, 15),
        title="test cameras",
    )

    # plt.show()
    plt.savefig(
        os.path.join("plots", f"{dataset_name}_test_cameras.png"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()
    
    print("done")