import sys
import os
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


def main(dataset_name, device):

    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        DATASETS_PATH,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
    )

    # random camera index
    rand_idx = 2  # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["train"][rand_idx])

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
    plot_points_2d_on_image(
        camera,
        points_2d_screen,  # [:, [1, 0]],
        show_ticks=True,
        figsize=(15, 15),
        title="screen space sampling (jittered)",
        show=False,
        save_path=os.path.join(
            "plots", f"{dataset_name}_screen_space_sampling_jittered.png"
        ),
    )

    # gen rays
    rays_o, rays_d, points_2d_screen = camera.get_rays(jitter_pixels=False)

    plot_points_2d_on_image(
        camera,
        points_2d_screen,  # [:, [1, 0]],
        show_ticks=True,
        figsize=(15, 15),
        title="screen space sampling",
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_screen_space_sampling.png"),
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
