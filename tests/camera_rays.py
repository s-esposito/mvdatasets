import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.utils.plotting import plot_cameras
from mvdatasets.utils.raycasting import get_camera_rays, get_camera_frames
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.common import get_dataset_test_preset

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

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])

    # resize camera
    taget_dim = 100
    min_dim = min(camera.width, camera.height)
    print("min_dim", min_dim)
    subsample_factor = min_dim // taget_dim
    print("subsample_factor", subsample_factor)
    camera.resize(subsample_factor=subsample_factor)

    print(camera)

    # gen rays
    rays_o, rays_d, points_2d = get_camera_rays(camera)

    vals, _ = get_camera_frames(camera, points_2d=points_2d)
    for key, val in vals.items():
        print(key, val.shape, val.device)

    # Visualize cameras
    fig = plot_cameras(
        [mv_data["test"][0]],
        points_3d=point_cloud,
        nr_rays=512,
        azimuth_deg=20,
        elevation_deg=30,
        up="z",
        draw_image_planes=True,
        draw_cameras_frustums=False,
        figsize=(15, 15),
        title=f"test camera {0} rays",
    )

    # plt.show()
    plt.savefig(
        os.path.join("plots", f"{dataset_name}_camera_rays.png"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
        transparent=True
    )
    plt.close()

    # # plt.show()
    # img_path = os.path.join("plots", f"{dataset_name}_camera_test_{rand_idx}.png")
    # img = camera.get_rgb()
    # mask = camera.get_mask()

    # # concatenate mask 3 times
    # mask = np.concatenate([mask] * 3, axis=-1)
    # print("mask", mask.shape)

    # # save image
    # plt.imsave(img_path, img)
    # plt.imsave(img_path.replace(".png", "_mask.png"), mask)