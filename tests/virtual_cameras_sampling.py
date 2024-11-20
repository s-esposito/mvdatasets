import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import DATASETS_PATH, DEVICE, SEED
from config import get_dataset_test_preset

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.visualization.matplotlib import plot_cameras
from mvdatasets.visualization.matplotlib import plot_current_batch
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.virtual_cameras import sample_cameras_on_hemisphere
from mvdatasets.geometry.primitives.bounding_box import BoundingBox


def main(dataset_name, device):

    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code

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

    # create bounding boxes
    bounding_boxes = []

    bb = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.scene_radius * 2,
        line_width=2.0,
        device=device,
    )
    bounding_boxes.append(bb)

    # only available for object centric datasets
    if not mv_data.cameras_on_hemisphere:
        exit(0)

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        point_cloud = None

    intrinsics = mv_data["train"][0].get_intrinsics()
    width = mv_data["train"][0].width
    height = mv_data["train"][0].height
    camera_center = mv_data["train"][0].get_center()
    camera_radius = np.linalg.norm(camera_center)

    sampled_cameras = sample_cameras_on_hemisphere(
        intrinsics=intrinsics,
        width=width,
        height=height,
        radius=camera_radius,
        nr_cameras=100,
    )

    # Visualize cameras
    plot_cameras(
        sampled_cameras,
        points_3d=point_cloud,
        azimuth_deg=20,
        elevation_deg=30,
        scene_radius=mv_data.max_camera_distance,
        up="z",
        figsize=(15, 15),
        title="sampled cameras",
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_sampled_cameras.png"),
    )

    # Create tensor reel
    tensor_reel = TensorReel(sampled_cameras, width=width, height=height, device=device)

    benchmark = False
    batch_size = 512
    nr_iterations = 10
    cameras_idx = None
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
            _,
            _,
        ) = tensor_reel.get_next_rays_batch(
            batch_size=batch_size,
            cameras_idx=cameras_idx,
        )

        if profiler is not None:
            profiler.end("get_next_rays_batch")

        if not benchmark:

            print("camera_idx", camera_idx.shape, camera_idx.device)
            print("rays_o", rays_o.shape, rays_o.device)
            print("rays_d", rays_d.shape, rays_d.device)

            plot_current_batch(
                sampled_cameras,
                camera_idx,
                rays_o,
                rays_d,
                rgb=None,
                mask=None,
                bounding_boxes=bounding_boxes,
                azimuth_deg=azimuth_deg,
                elevation_deg=30,
                scene_radius=mv_data.max_camera_distance,
                up="z",
                figsize=(15, 15),
                show=False,
                save_path=os.path.join(
                    "plots", f"{dataset_name}_sampled_cameras_batch_{i}.png"
                ),
            )

            # update azimuth every 2 iterations
            if i % 2 != 0:
                azimuth_deg += azimuth_deg_delta

    if profiler is not None:
        profiler.print_avg_times()


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
        dataset_name = "blender"

    main(dataset_name, DEVICE)
