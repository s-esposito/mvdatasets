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
        verbose=True,
    )

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])
    print(camera)

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        # dataset has not tests/assets point cloud
        exit(0)

    points_2d_s = camera.project_points_3d_to_2d(points_3d=point_cloud)
    print("points_2d_s", points_2d_s.shape)

    # 3d points distance from camera center
    camera_points_dists = camera.camera_to_points_3d_distance(point_cloud)
    print("camera_points_dist", camera_points_dists.shape)

    fig = plot_points_2d_on_image(camera, points_2d_s, points_norms=camera_points_dists)

    # plt.show()
    plt.savefig(
        os.path.join("plots", f"{dataset_name}_point_cloud_projection.png"),
        transparent=True,
        dpi=300,
    )
    plt.close()

    # reproject to 3D

    # filter out points outside image range
    points_mask = points_2d_s[:, 0] >= 0
    points_mask *= points_2d_s[:, 1] >= 0
    points_mask *= points_2d_s[:, 0] < camera.width
    points_mask *= points_2d_s[:, 1] < camera.height
    points_2d_s = points_2d_s[points_mask]
    camera_points_dists = camera_points_dists[points_mask]
    point_cloud = point_cloud[points_mask]

    points_3d = camera.unproject_points_2d_to_3d(
        points_2d_s=points_2d_s, depth=camera_points_dists
    )

    # filter out random number of points
    num_points = 100
    idx = np.random.choice(range(len(points_3d)), num_points, replace=False)
    point_cloud = point_cloud[idx]
    points_3d = points_3d[idx]

    # visualize 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c="r",
        marker="o",
        label="point cloud",
    )
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        c="b",
        marker="x",
        label="reprojected points",
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.legend()

    error = np.mean(np.abs(points_3d - point_cloud))
    print("error", error.item())

    # plt.show()
    plt.savefig(
        os.path.join("plots", f"{dataset_name}_point_cloud_unprojection.png"),
        transparent=True,
        dpi=300,
    )
    plt.close()
