import os
import sys
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.utils.plotting import plot_current_batch
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.profiler import Profiler

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
dataset_name = "dtu"
scene_name = "dtu_scan83"

# dataset loading
mv_data = MVDataset(
    dataset_name,
    scene_name,
    datasets_path,
    splits=["train", "test"]
)

batch_size = 512

# TensorReel (~1300 it/s), camera's data in concatenated in big tensors on GPU

tensor_reel = TensorReel(mv_data["train"], device=device)

nr_iterations = 10
cameras_idx = np.array([0])  # None
frame_idx = None
pbar = tqdm(range(nr_iterations))
azimuth_deg = 0
azimuth_deg_delta = 360 / (nr_iterations / 2)
frames_paths = []
for i in pbar:
    pbar.set_description("ray casting")

    # cameras_idx = np.random.permutation(len(mv_data["train"]))[:1]

    if profiler is not None:
        profiler.start("get_next_batch")

    # get rays and gt values
    with torch.set_grad_enabled(False):
        (
            camera_idx,
            rays_o,
            rays_d,
            gt_rgb,
            gt_mask,
            frame_idx,
        ) = tensor_reel.get_next_batch(
            batch_size=batch_size, cameras_idx=cameras_idx, frame_idx=frame_idx
        )

    if profiler is not None:
        profiler.end("get_next_batch")

    fig = plot_current_batch(
        mv_data["train"],
        camera_idx,
        rays_o,
        rays_d,
        gt_rgb,
        gt_mask,
        azimuth_deg=azimuth_deg,
        elevation_deg=30,
        up="y",
        figsize=(15, 15),
    )

    # plt.show()
    frame_path = os.path.join("plots", f"test_static_scenes_batch_{i}.png")
    plt.savefig(
        frame_path,
        # bbox_inches="tight",
        # pad_inches=0,
        dpi=300
    )
    plt.close()
    frames_paths.append(frame_path)

    # update azimuth every 2 iterations
    if i % 2 != 0:
        azimuth_deg += azimuth_deg_delta

# # make webm video
# video_path = os.path.join("plots", "test.webm")
# writer = imageio.get_writer(video_path, fps=30, codec="libvpx-vp9")
# for frame_path in frames_paths:
#     im = imageio.imread(frame_path)
#     writer.append_data(im)
#     os.remove(frame_path)

# for i in tqdm(range(nr_iterations)):
#     profiler.start("get_next_batch")
#     idx, rays_o, rays_d, rgb, mask = get_next_batch(data_loader)
#     profiler.end("get_next_batch")

#     fig = plot_current_batch(
#         dataset_train.cameras,
#         idx,
#         rays_o,
#         rays_d,
#         rgb,
#         mask,
#         azimuth_deg=60,
#         elevation_deg=30,
#         up="y",
#         figsize=(15, 15),
#     )

#     # plt.show()
#     plt.savefig(f"test_static_scenes_batch_{i}.png", bbox_inches="tight", pad_inches=0, dpi=300)

if profiler is not None:
    profiler.print_avg_times()