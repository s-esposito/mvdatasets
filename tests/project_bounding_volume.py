import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.utils.plotting import plot_points_2d_on_image
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.utils.raycasting import get_camera_rays
from mvdatasets.utils.bounding_box import BoundingBox
from mvdatasets.utils.bounding_sphere import BoundingSphere
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
    
    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])
    # shoot rays from camera
    rays_o, rays_d, points_2d = get_camera_rays(camera, device=device)
    
    # bounding box
    scene_radius = mv_data.scene_radius
    scene_diameter = scene_radius * 2
    bounding_volume = BoundingBox(
        pose=np.eye(4),
        local_scale=np.array([scene_diameter, scene_diameter, scene_diameter]),
        device=device
    )
    # bounding_volume intersection test
    is_hit, t_near, t_far, p_near, p_far = bounding_volume.intersect(rays_o, rays_d)
    hit_range = (t_far - t_near).reshape(camera.height, camera.width)
    hit_range = hit_range.cpu().numpy()
    # get the color map
    color_map = plt.colormaps.get_cmap("jet")
    # apply the colormap
    hit_range = color_map(hit_range)[:, :, :3]
    # print("hit range", hit_range)
    
    img_np = ((camera.get_rgb() / 255.0) * 0.5) + (hit_range * 0.5)
    plt.imshow(img_np)
    
    plt.savefig(os.path.join("plots", f"{dataset_name}_bounding_box.png"), transparent=True, dpi=300)
    plt.close()
    
    # bounding sphere
    bounding_volume = BoundingSphere(
        pose=np.eye(4),
        local_scale=scene_radius,
        device=device
    )
    # bounding_volume intersection test
    is_hit, t_near, t_far, p_near, p_far = bounding_volume.intersect(rays_o, rays_d)
    hit_range = (t_far - t_near).reshape(camera.height, camera.width)
    hit_range = hit_range.cpu().numpy()
    # get the color map
    color_map = plt.colormaps.get_cmap("jet")
    # apply the colormap
    hit_range = color_map(hit_range)[:, :, :3]
    # print("hit range", hit_range)
    
    img_np = ((camera.get_rgb() / 255.0) * 0.5) + (hit_range * 0.5)
    plt.imshow(img_np)
    
    plt.savefig(os.path.join("plots", f"{dataset_name}_bounding_sphere.png"), transparent=True, dpi=300)
    plt.close()
    
    # points_3d = bounding_volume.get_random_points_on_surface(100000)
    # points_2d = camera.project_points_3d_to_2d(points_3d=points_3d)
    # # 3d points distance from camera center
    # camera_points_dists = camera.camera_to_points_3d_distance(points_3d)
    # points_2d = points_2d.cpu().numpy()
    # camera_points_dists = camera_points_dists.cpu().numpy()
    # print("camera_points_dist", camera_points_dists)
    # fig = plot_points_2d_on_image(camera, points_2d, points_norms=camera_points_dists)
    # plt.show()