import tyro
import sys
import os
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from examples import get_dataset_test_preset
from examples import Args
from mvdatasets.visualization.matplotlib import plot_camera_2d
from mvdatasets.mvdataset import MVDataset


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    scene_name = args.scene_name
    test_preset = get_dataset_test_preset(dataset_name)
    if scene_name is None:
        scene_name = test_preset["scene_name"]
    pc_paths = test_preset["pc_paths"]
    splits = test_preset["splits"]

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=splits,
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
        show=False,
        save_path=os.path.join(
            "plots", f"{dataset_name}_{scene_name}_screen_space_sampling_jittered.png"
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
        show=False,
        save_path=os.path.join(
            "plots", f"{dataset_name}_{scene_name}_screen_space_sampling.png"
        ),
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
