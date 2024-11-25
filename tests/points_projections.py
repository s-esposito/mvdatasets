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
from mvdatasets.visualization.matplotlib import plot_points_2d_on_image, plot_3d
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
    rand_idx = torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])
    print(camera)

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        # dataset has not tests/assets point cloud
        print_error("No point cloud found in the dataset")

    points_2d_screen, points_mask = camera.project_points_3d_world_to_2d_screen(
        points_3d=point_cloud, filter_points=True
    )
    print("points_2d_screen", points_2d_screen.shape)

    point_cloud = point_cloud[points_mask]

    # 3d points distance from camera center
    camera_points_dists = camera.distance_to_points_3d_world(point_cloud)
    print("camera_points_dist", camera_points_dists.shape)

    plot_points_2d_on_image(
        camera,
        points_2d_screen,
        title="point cloud projection",
        points_norms=camera_points_dists,
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_point_cloud_projection.png"),
    )

    # reproject to 3D
    points_3d = camera.unproject_points_2d_screen_to_3d_world(
        points_2d_screen=points_2d_screen, depth=camera_points_dists
    )

    # filter out random number of points
    num_points = 100
    idx = np.random.choice(range(len(points_3d)), num_points, replace=False)
    point_cloud = point_cloud[idx]
    points_3d = points_3d[idx]

    # plot point clouds and camera
    plot_3d(
        cameras=[camera],
        points_3d=[point_cloud, points_3d],
        points_3d_colors=["red", "blue"],
        points_3d_labels=["point cloud", "unprojected points"],
        points_3d_markers=["o", "x"],
        points_3d_sizes=[150, 100],
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

    error = np.mean(np.abs(points_3d - point_cloud))
    print("error", error.item())


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
