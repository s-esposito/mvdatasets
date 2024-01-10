import sys
import os
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.utils.plotting import plot_points_2d_on_image
from mvdatasets.utils.raycasting import get_camera_rays
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.common import get_dataset_test_preset

if __name__ == "__main__":

    # Set a random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    # # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(seed)  # Set a random seed for GPU
    else:
        device = "cuda"
    torch.set_default_device(device)

    # Set default tensor type
    torch.set_default_dtype(torch.float32)

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code

    # Set datasets path
    datasets_path = "/home/stefano/Data"

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
        splits=["train", "test"]
    )

    # random camera index
    rand_idx = 0 # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])
    print(camera)

    # resize camera's rgb modality
    camera.resize(max_dim=100)

    # gen rays
    rays_o, rays_d, points_2d = get_camera_rays(camera, jitter_pixels=True)
    fig = plot_points_2d_on_image(
        camera,
        points_2d[:, [1, 0]],
        show_ticks=True,
        figsize=(15, 15)
    )
    plt.savefig(
        os.path.join("imgs", f"{dataset_name}_screen_space_sampling_jittered.png"),
        transparent=True,
        dpi=300
    )
    plt.close()

    # gen rays
    rays_o, rays_d, points_2d = get_camera_rays(camera, jitter_pixels=False)

    fig = plot_points_2d_on_image(
        camera,
        points_2d[:, [1, 0]],
        show_ticks=True,
        figsize=(15, 15)
    )
    plt.savefig(
        os.path.join("imgs", f"{dataset_name}_screen_space_sampling.png"),
        transparent=True,
        dpi=300
    )
    plt.close()