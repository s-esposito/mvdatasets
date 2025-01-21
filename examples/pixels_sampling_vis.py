import tyro
import sys
import os
import torch
from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mvdatasets.visualization.matplotlib import plot_camera_2d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.configs.example_config import ExampleConfig
from examples import get_dataset_test_preset, custom_exception_handler


def main(cfg: ExampleConfig, pc_paths: list[Path]):

    device = cfg.machine.device
    datasets_path = cfg.datasets_path
    output_path = cfg.output_path
    scene_name = cfg.scene_name
    dataset_name = cfg.data.dataset_name

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        config=cfg.data,
        point_clouds_paths=pc_paths,
        verbose=True,
    )

    # random camera index
    rand_idx = 2  # torch.randint(0, len(mv_data.get_split("test")), (1,)).item()
    camera = deepcopy(mv_data.get_split("train")[rand_idx])

    # resize camera
    taget_dim = 100
    min_dim = min(camera.width, camera.height)
    print("min_dim", min_dim)
    subsample_factor = min_dim // taget_dim
    print("subsample_factor", subsample_factor)
    camera.resize(subsample_factor=subsample_factor)

    print(camera)

    # gen rays
    rays_o, rays_d, points_2d_screen = camera.get_rays(jitter_pixels=True)
    plot_camera_2d(
        camera,
        points_2d_screen,
        show_ticks=True,
        figsize=(15, 15),
        title="screen space sampling (jittered)",
        show=cfg.with_viewer,
        save_path=os.path.join(
            output_path,
            f"{dataset_name}_{scene_name}_screen_space_sampling_jittered.png",
        ),
    )

    # gen rays
    rays_o, rays_d, points_2d_screen = camera.get_rays(jitter_pixels=False)
    plot_camera_2d(
        camera,
        points_2d_screen,
        show_ticks=True,
        figsize=(15, 15),
        title="screen space sampling",
        show=cfg.with_viewer,
        save_path=os.path.join(
            output_path, f"{dataset_name}_{scene_name}_screen_space_sampling.png"
        ),
    )


if __name__ == "__main__":

    # custom exception handler
    sys.excepthook = custom_exception_handler

    # parse arguments
    args = tyro.cli(ExampleConfig)

    # get test preset
    test_preset = get_dataset_test_preset(args.data.dataset_name)
    # scene name
    if args.scene_name is None:
        args.scene_name = test_preset["scene_name"]
        print_warning(
            f"scene_name is None, using preset test scene {args.scene_name} for dataset"
        )
    # additional point clouds paths (if any)
    pc_paths = test_preset["pc_paths"]

    # start the example program
    main(args, pc_paths)
