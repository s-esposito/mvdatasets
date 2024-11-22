import tyro
import sys
import os
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
from config import Args
from config import get_dataset_test_preset
from mvdatasets.visualization.matplotlib import plot_points_2d_on_image
from mvdatasets.utils.raycasting import (
    get_points_2d_screen_from_pixels,
    get_random_pixels,
    get_random_pixels_from_error_map,
)
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.images import image_to_tensor


def main(args: Args):

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    if dataset_name != "plushy":
        raise ValueError("This test is only for the plushy dataset.")

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True,
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
    for _ in range(10000):
        profiler.start("uniform_sampling")
        pixels = get_random_pixels(
            camera.height, camera.width, nr_rays, device=device
        )  # (W, H, 2)
        points_2d_screen = get_points_2d_screen_from_pixels(
            pixels, jitter_pixels
        )  # (N, 2)
        rays_o, rays_d, points_2d_screen = camera.get_rays(
            points_2d_screen=points_2d_screen, device=device
        )
        profiler.end("uniform_sampling")

    points_2d_screen = points_2d_screen.cpu().numpy()

    plot_points_2d_on_image(
        camera,
        points_2d_screen,  # [:, [1, 0]],
        show_ticks=False,
        figsize=(15, 15),
        title="uniform sampling",
        show=False,
        save_path=os.path.join("plots", "ray_sampling_uniform.png"),
    )

    # load error map
    error_map_pil = Image.open("tests/assets/error_maps/plushy_test_0.png")
    error_map = image_to_tensor(error_map_pil, device=device)

    # gen rays (biased sampling)
    for i in range(10000):
        profiler.start("error_map_sampling")
        pixels = get_random_pixels_from_error_map(
            error_map, nr_rays, device=device
        )  # (N, 2)
        points_2d_screen = get_points_2d_screen_from_pixels(
            pixels, jitter_pixels
        )  # (N, 2)
        rays_o, rays_d, points_2d_screen = camera.get_rays(
            points_2d_screen=points_2d_screen, device=device
        )
        profiler.end("error_map_sampling")

    points_2d_screen = points_2d_screen.cpu().numpy()

    plot_points_2d_on_image(
        camera,
        points_2d_screen,  # [:, [1, 0]],
        show_ticks=False,
        figsize=(15, 15),
        title="error map sampling",
        show=False,
        save_path=os.path.join("plots", "ray_sampling_error_proportional.png"),
    )

    profiler.print_avg_times()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
