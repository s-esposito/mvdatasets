import sys
import os
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.utils.plotting import plot_points_2d_on_image
from mvdatasets.utils.raycasting import (
    get_camera_rays,
    get_points_2d_from_pixels,
    get_random_pixels,
    get_random_pixels_from_error_map,
)
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.utils.images import numpy_to_image, image_to_tensor
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

    dataset_name = "blendernerf"
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
    )

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])

    # # resize camera
    # taget_dim = 100
    # min_dim = min(camera.width, camera.height)
    # print("min_dim", min_dim)
    # subsample_factor = min_dim // taget_dim
    # print("subsample_factor", subsample_factor)
    # camera.resize(subsample_factor=subsample_factor)

    print(camera)

    jitter_pixels = False
    nr_rays = 4096

    # gen rays (uniform sampling)
    for i in range(10000):
        profiler.start("uniform_sampling")
        pixels = get_random_pixels(camera.height, camera.width, nr_rays, device=device)
        points_2d = get_points_2d_from_pixels(
            pixels, jitter_pixels, camera.height, camera.width
        )
        rays_o, rays_d, points_2d = get_camera_rays(
            camera, points_2d=points_2d, device=device
        )
        profiler.end("uniform_sampling")

    points_2d = points_2d.cpu().numpy()

    fig = plot_points_2d_on_image(
        camera, points_2d[:, [1, 0]], show_ticks=False, figsize=(15, 15)
    )
    plt.savefig(
        os.path.join("plots", "ray_sampling_uniform.png"), transparent=True, dpi=300
    )
    plt.close()

    # load error map
    error_map_pil = Image.open("tests/assets/error_maps/plushy_test_0.png")
    error_map = image_to_tensor(error_map_pil, device=device)

    # gen rays (uniform sampling)
    for i in range(10000):
        profiler.start("error_map_sampling")
        pixels = get_random_pixels_from_error_map(
            error_map, camera.height, camera.width, nr_rays, device=device
        )

        points_2d = get_points_2d_from_pixels(
            pixels, jitter_pixels, camera.height, camera.width
        )
        rays_o, rays_d, points_2d = get_camera_rays(
            camera, points_2d=points_2d, device=device
        )
        profiler.end("error_map_sampling")

    points_2d = points_2d.cpu().numpy()

    fig = plot_points_2d_on_image(
        camera, points_2d[:, [1, 0]], show_ticks=False, figsize=(15, 15)
    )
    plt.savefig(
        os.path.join("plots", "ray_sampling_error_proportional.png"),
        transparent=True,
        dpi=300,
    )
    plt.close()

    profiler.print_avg_times()
