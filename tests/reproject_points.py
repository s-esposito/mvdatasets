import os
import sys
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.utils.plotting import plot_points_2d_on_image
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.geometry import project_points_3d_to_2d

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

datasets_path = "/home/stefano/Data"
# dataset_names = ["dtu", "blender"]
# scene_names = ["dtu_scan83", "lego"]
dataset_names = ["blender"]
scene_names = ["lego"]

for dataset_name, scene_name in zip(dataset_names, scene_names):

    # load gt point cloud if exists
    gt_point_clouds_paths = [os.path.join("debug/point_clouds/", dataset_name, f"{scene_name}.ply")]

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=gt_point_clouds_paths,
        splits=["train", "test"]
    )

    # random camera index
    rand_idx = torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])
    print(camera)

    point_cloud = mv_data.point_clouds[0]
    points_2d = project_points_3d_to_2d(camera=camera, points_3d=point_cloud)

    fig = plot_points_2d_on_image(camera, points_2d)

    # plt.show()
    plt.savefig(os.path.join("imgs", f"{dataset_name}_point_cloud_projection.png"), transparent=True, dpi=300)
    plt.close()