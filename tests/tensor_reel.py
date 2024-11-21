import tyro
import os
import sys
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import get_dataset_test_preset
from config import Args

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.visualization.matplotlib import plot_current_batch
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.profiler import Profiler
from mvdatasets.geometry.primitives.bounding_box import BoundingBox


def main(args: Args):

    device = args.device
    scene_name, pc_paths, config = get_dataset_test_preset(args.dataset_name)

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code

    # dataset loading
    mv_data = MVDataset(
        args.dataset_name,
        scene_name,
        args.datasets_path,
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
        )

        if profiler is not None:
            profiler.end("get_next_rays_batch")

        if not benchmark:

            gt_rgb = vals["rgbs"]
            if "masks" in vals:
                gt_mask = vals["masks"]
            else:
                gt_mask = None

            print("camera_idx", camera_idx.shape, camera_idx.device, camera_idx.dtype)
            print("rays_o", rays_o.shape, rays_o.device, rays_o.dtype)
            print("rays_d", rays_d.shape, rays_d.device, rays_d.dtype)
            print("gt_rgb", gt_rgb.shape, gt_rgb.device, gt_rgb.dtype)
            if gt_mask is not None:
                print("gt_mask", gt_mask.shape, gt_mask.device, gt_mask.dtype)
            print("frame_idx", frame_idx.shape, frame_idx.device, frame_idx.dtype)

            plot_current_batch(
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
                show=False,
                save_path=os.path.join("plots", f"{dataset_name}_batch_{i}.png"),
            )

            # update azimuth every 2 iterations
            if i % 2 != 0:
                azimuth_deg += azimuth_deg_delta

    if profiler is not None:
        profiler.print_avg_times()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)