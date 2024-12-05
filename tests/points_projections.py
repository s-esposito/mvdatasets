import tyro
from rich import print
import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from config import get_dataset_test_preset
from config import Args
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.visualization.matplotlib import plot_camera_2d, plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.printing import print_error


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    test_preset = get_dataset_test_preset(dataset_name)
    scene_name = test_preset["scene_name"]
    pc_paths = test_preset["pc_paths"]
    config = test_preset["config"]
    splits = test_preset["splits"]

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=splits,
        config=config,
        verbose=True,
    )

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data.get_split("test")), (1,)).item()
    camera = mv_data.get_split("test")[rand_idx]
    
    # random frame index
    frame_idx = torch.randint(0, camera.temporal_dim, (1,)).item()

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        # dataset has not tests/assets point cloud
        print_error("No point cloud found in the dataset")

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
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_point_cloud_projection.png"),
    )

    # reproject to 3D
    points_3d_r = camera.unproject_points_2d_screen_to_3d_world(
        points_2d_screen=points_2d_screen, depth=camera_points_dists
    )

    # filter out random number of points
    num_points = 100
    idx = np.random.choice(range(len(points_3d)), num_points, replace=False)
    points_3d = points_3d[idx]
    points_3d_r = points_3d_r[idx]

    # create new point clouds for visualization
    pcs = []
    pcs.append(
        PointCloud(
            points_3d=points_3d,
            color="red",
            label="point cloud",
            marker="o",
            size=150,
        )
    )
    pcs.append(
        PointCloud(
            points_3d=points_3d_r,
            color="blue",
            label="reprojected points",
            marker="x",
            size=100,
        )
    )

    # plot point clouds and camera
    plot_3d(
        cameras=[camera],
        point_clouds=pcs,
        azimuth_deg=20,
        elevation_deg=30,
        up="z",
        scene_radius=mv_data.get_scene_radius(),
        draw_bounding_cube=True,
        draw_image_planes=True,
        figsize=(15, 15),
        title="point cloud unprojection",
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_point_cloud_unprojection.png"),
    )

    error = np.mean(np.abs(points_3d_r - points_3d))
    print("error", error.item())


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
