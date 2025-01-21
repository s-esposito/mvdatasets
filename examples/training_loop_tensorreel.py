import tyro
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from mvdatasets.visualization.matplotlib import plot_current_batch
from mvdatasets.mvdataset import MVDataset
from mvdatasets.tensorreel import TensorReel
from mvdatasets.utils.profiler import Profiler
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.configs.example_config import ExampleConfig
from mvdatasets.utils.printing import print_warning
from examples import get_dataset_test_preset, custom_exception_handler


def main(
    cfg: ExampleConfig,
    pc_paths: list[Path],
    splits: list[str]
):

    device = cfg.machine.device
    datasets_path = cfg.datasets_path
    output_path = cfg.output_path
    scene_name = cfg.scene_name
    dataset_name = cfg.data.dataset_name

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        splits=splits,
        config=cfg.data,
        point_clouds_paths=pc_paths,
        verbose=True,
    )

    # create bounding box
    bb = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.get_foreground_radius() * 2,
        line_width=2.0,
        device=device,
    )

    # TensorReel (~1300 it/s), camera's data in concatenated in big tensors on GPU

    tensorreel = TensorReel(mv_data.get_split("train"), device=device)
    print(tensorreel)

    batch_size = 512

    benchmark = False

    if benchmark:
        # Set profiler
        profiler = Profiler()  # nb: might slow down the code
        nr_iterations = 10000
    else:
        profiler = None
        nr_iterations = 10

    # use a subset of cameras and frames
    # cameras_idx = np.random.permutation(len(mv_data.get_split("train")))[:5]
    # frames_idx = np.random.permutation(mv_data.get_nr_per_camera_frames())[:2]
    # or use all
    cameras_idx = None
    frames_idx = None
    print("cameras_idx", cameras_idx)
    print("frames_idx", frames_idx)

    # -------------------------------------------------------------------------

    pbar = tqdm(range(nr_iterations), desc="ray casting", ncols=100)
    azimuth_deg = 20
    azimuth_deg_delta = 1
    for i in pbar:

        if profiler is not None:
            profiler.start("get_next_rays_batch")

        # get rays and gt values
        batch = tensorreel.get_next_rays_batch(
            batch_size=batch_size,
            cameras_idx=cameras_idx,
            frames_idx=frames_idx,
            jitter_pixels=True,
            nr_rays_per_pixel=1,
        )

        if profiler is not None:
            profiler.end("get_next_rays_batch")

        if not benchmark:

            # unpack batch
            batch_cameras_idx = batch["cameras_idx"]
            batch_rays_o = batch["rays_o"]
            batch_rays_d = batch["rays_d"]
            batch_vals = batch["vals"]
            
            # print data shapes
            for k, v in batch.items():
                # if v is a dict
                if isinstance(v, dict):
                    for k1, v1 in v.items():
                        print(f"{k}, " f"{k1}", v1.shape, v1.device, v1.dtype)
                else:
                    print(f"{k}", v.shape, v.device, v.dtype)

            # get gt values
            gt_rgb = batch_vals["rgbs"]
            if "masks" in batch_vals:
                gt_mask = batch_vals["masks"]
            else:
                gt_mask = None

            plot_current_batch(
                cameras=mv_data.get_split("train"),
                cameras_idx=batch_cameras_idx.cpu().numpy(),
                rays_o=batch_rays_o.cpu().numpy(),
                rays_d=batch_rays_d.cpu().numpy(),
                rgbs=gt_rgb.cpu().numpy(),
                masks=gt_mask.cpu().numpy() if gt_mask is not None else None,
                bounding_boxes=[bb],
                azimuth_deg=azimuth_deg,
                elevation_deg=30,
                scene_radius=mv_data.get_scene_radius(),
                up="z",
                figsize=(15, 15),
                title=f"rays batch sampling {i}",
                show=cfg.with_viewer,
                save_path=os.path.join(
                    output_path, f"{dataset_name}_{scene_name}_batch_{i}.png"
                ),
            )

            # update azimuth
            azimuth_deg += azimuth_deg_delta

    if profiler is not None:
        profiler.print_avg_times()


if __name__ == "__main__":
    
    # custom exception handler
    sys.excepthook = custom_exception_handler
    
    # parse arguments
    args = tyro.cli(ExampleConfig)
    
    # get test preset
    test_preset = get_dataset_test_preset(args.data.dataset_name)
    # scene name
    if args.scene_name is None:
        args.scene_name = test_preset["scene_name"]
        print_warning(f"scene_name is None, using preset test scene {args.scene_name} for dataset")
    # additional point clouds paths (if any)
    pc_paths = test_preset["pc_paths"]
    # testing splits
    splits = test_preset["splits"]

    # start the example program
    main(args, pc_paths, splits)