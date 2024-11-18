import os
import sys
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.visualization.matplotlib import plot_current_batch
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.profiler import Profiler
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox


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
        splits=["train", "test"],
        config=config,
        verbose=True,
    )

    # create bounding boxes
    bounding_boxes = []

    bb = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.scene_radius * 2,
        line_width=2.0,
        device=device,
    )
    bounding_boxes.append(bb)

    # TensorReel (~1300 it/s), camera's data in concatenated in big tensors on GPU

    tensor_reel = TensorReel(mv_data["train"], device=device)
    print(tensor_reel)

    benchmark = False
    batch_size = 512
    nr_iterations = 10
    cameras_idx = None
    frame_idx = None
    pbar = tqdm(range(nr_iterations), desc="ray casting", ncols=100)
    azimuth_deg = 0
    azimuth_deg_delta = 360 / (nr_iterations / 2)
    frames_paths = []
    for i in pbar:

        # cameras_idx = np.random.permutation(len(mv_data["train"]))[:2]

        if profiler is not None:
            profiler.start("get_next_rays_batch")

        # get rays and gt values
        (
            camera_idx,
            rays_o,
            rays_d,
            vals,
            frame_idx,
        ) = tensor_reel.get_next_rays_batch(
            batch_size=batch_size,
            cameras_idx=cameras_idx,
            frames_idx=frame_idx,
            jitter_pixels=True,
            nr_rays_per_pixel=1,
            # masked_sampling=True
        )

        if profiler is not None:
            profiler.end("get_next_rays_batch")

        if not benchmark:

            gt_rgb = vals["rgb"]
            if "mask" in vals:
                gt_mask = vals["mask"]
            else:
                gt_mask = None

            print("camera_idx", camera_idx.shape, camera_idx.device, camera_idx.dtype)
            print("rays_o", rays_o.shape, rays_o.device, rays_o.dtype)
            print("rays_d", rays_d.shape, rays_d.device, rays_d.dtype)
            print("gt_rgb", gt_rgb.shape, gt_rgb.device, gt_rgb.dtype)
            if gt_mask is not None:
                print("gt_mask", gt_mask.shape, gt_mask.device, gt_mask.dtype)
            print("frame_idx", frame_idx.shape, frame_idx.device, frame_idx.dtype)

            fig = plot_current_batch(
                mv_data["train"],
                camera_idx,
                rays_o,
                rays_d,
                rgb=gt_rgb,
                mask=gt_mask,
                bounding_boxes=bounding_boxes,
                azimuth_deg=azimuth_deg,
                elevation_deg=30,
                scene_radius=mv_data.max_camera_distance,
                up="z",
                figsize=(15, 15),
            )

            # plt.show()
            frame_path = os.path.join("plots", f"{dataset_name}_batch_{i}.png")
            plt.savefig(
                frame_path, bbox_inches="tight", pad_inches=0, dpi=72, transparent=True
            )
            plt.close()
            frames_paths.append(frame_path)

            # update azimuth every 2 iterations
            if i % 2 != 0:
                azimuth_deg += azimuth_deg_delta

    if profiler is not None:
        profiler.print_avg_times()
