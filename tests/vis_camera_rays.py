import tyro
import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from config import get_dataset_test_preset
from config import Args

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.visualization.matplotlib import plot_3d, plot_image
from mvdatasets.mvdataset import MVDataset


def main(args: Args):

    device = args.device
    dataset_path = args.datasets_path
    dataset_name = args.dataset_name
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        dataset_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True,
    )

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        point_cloud = None

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])

    # resize camera
    camera.resize(subsample_factor=10)

    # print(camera)

    # gen rays and get data
    # rays_o, rays_d, points_2d_screen = camera.get_rays()
    # vals = camera.get_data(points_2d_screen=points_2d_screen)

    # vals = camera.get_data()
    # for key, val in vals.items():
    #     if val is not None:
    #         val = val.reshape(camera.width, camera.height, -1).cpu().numpy()
    #         # plot
    #         plot_image(
    #             image=val,
    #             show=True,
    #         )

    # Visualize cameras
    plot_3d(
        cameras=[camera],
        points_3d=point_cloud,
        nr_rays=256,
        azimuth_deg=20,
        elevation_deg=30,
        scene_radius=1.0,
        up="z",
        draw_image_planes=True,
        draw_cameras_frustums=False,
        figsize=(15, 15),
        title=f"test camera {rand_idx} rays",
        show=True,
        save_path=os.path.join("plots", f"{dataset_name}_camera_rays.png"),
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)