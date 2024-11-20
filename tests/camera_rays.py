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
from mvdatasets.visualization.matplotlib import plot_cameras
from mvdatasets.mvdataset import MVDataset


def main(dataset_name, device):

    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

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

    vals = camera.get_data()
    for key, val in vals.items():
        if val is not None:
            val = val.reshape(camera.height, camera.width, -1).cpu().numpy()
            plt.imshow(val)
            plt.xlabel("h")
            plt.ylabel("w")
            plt.show()
            print(key, val.shape)

    exit(0)

    # Visualize cameras
    plot_cameras(
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

    main(dataset_name, DEVICE)
