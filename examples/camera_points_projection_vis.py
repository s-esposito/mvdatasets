import tyro
from rich import print
import os
import sys
import torch
from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.visualization.matplotlib import plot_camera_2d, plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.printing import print_error, print_warning
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
    rand_idx = 0  # torch.randint(0, len(mv_data.get_split("train")), (1,)).item()
    camera = mv_data.get_split("train")[rand_idx]

    # random frame index
    frame_idx = torch.randint(0, camera.temporal_dim, (1,)).item()

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        # dataset has not examples/assets point cloud
        raise ValueError("No point cloud found in the dataset")

    points_3d = point_cloud.points_3d

    points_2d_screen, points_mask = camera.project_points_3d_world_to_2d_screen(
        points_3d=points_3d, filter_points=True
    )
    print("points_2d_screen", points_2d_screen.shape)

    points_3d = points_3d[points_mask]

    # 3d points distance from camera center
    camera_points_dists = camera.distance_to_points_3d_world(points_3d)
    print("camera_points_dist", camera_points_dists.shape)

    plot_camera_2d(
        camera,
        points_2d_screen,
        frame_idx=frame_idx,
        title="point cloud projection",
        points_norms=camera_points_dists,
        show=cfg.with_viewer,
        save_path=os.path.join(
            output_path, f"{dataset_name}_{scene_name}_point_cloud_projection.png"
        ),
    )

    # # reproject to 3D
    # points_3d_r = camera.unproject_points_2d_screen_to_3d_world(
    #     points_2d_screen=points_2d_screen, depth=camera_points_dists
    # )

    # # filter out random number of points
    # num_points = 100
    # if len(points_3d) > num_points:
    #     idx = np.random.choice(range(len(points_3d)), num_points, replace=False)
    #     points_3d = points_3d[idx]
    #     points_3d_r = points_3d_r[idx]

    # # create new point clouds for visualization
    # pcs = []
    # pcs.append(
    #     PointCloud(
    #         points_3d=points_3d,
    #         color="red",
    #         label="point cloud",
    #         marker="o",
    #         size=150,
    #     )
    # )
    # pcs.append(
    #     PointCloud(
    #         points_3d=points_3d_r,
    #         color="blue",
    #         label="reprojected points",
    #         marker="x",
    #         size=100,
    #     )
    # )

    # # plot point clouds and camera
    # plot_3d(
    #     cameras=[camera],
    #     point_clouds=pcs,
    #     azimuth_deg=20,
    #     elevation_deg=30,
    #     up="z",
    #     scene_radius=mv_data.get_scene_radius(),
    #     draw_bounding_cube=True,
    #     draw_image_planes=True,
    #     figsize=(15, 15),
    #     title="point cloud unprojection",
    #     show=cfg.with_viewer,
    #     save_path=os.path.join(
    #         output_path, f"{dataset_name}_{scene_name}_point_cloud_unprojection.png"
    #     ),
    # )

    # error = np.mean(np.abs(points_3d_r - points_3d))
    # print("error", error.item())


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
