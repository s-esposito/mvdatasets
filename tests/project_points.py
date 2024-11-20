from rich import print
import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from config import get_dataset_test_preset
from config import DATASETS_PATH, DEVICE, SEED

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.visualization.matplotlib import plot_points_2d_on_image
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.printing import print_error


def main(dataset_name, device):

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        DATASETS_PATH,
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


if __name__ == "__main__":

    # Set a random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)  # Set a random seed for GPU
    torch.set_default_device(DEVICE)

    # Set default tensor type
    torch.set_default_dtype(torch.float32)

    # Get dataset test preset

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "dtu"
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    main(dataset_name, DEVICE)
