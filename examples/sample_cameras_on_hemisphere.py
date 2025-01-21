import tyro
import numpy as np
import os
import sys
from tqdm import tqdm
from pathlib import Path
from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.visualization.matplotlib import plot_current_batch
from mvdatasets.visualization.matplotlib import plot_rays_samples
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.tensorreel import TensorReel
from mvdatasets.utils.virtual_cameras import sample_cameras_on_hemisphere
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.utils.printing import print_warning, print_error
from mvdatasets.configs.example_config import ExampleConfig
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

    bb = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.get_foreground_radius() * 2,
        device=device,
    )

    # only available for object centric datasets
    if not mv_data.cameras_on_hemisphere:
        raise ValueError(f"{dataset_name} does not have cameras on hemisphere")

    intrinsics = mv_data.get_split("train")[0].get_intrinsics()
    width = mv_data.get_split("train")[0].width
    height = mv_data.get_split("train")[0].height
    camera_center = mv_data.get_split("train")[0].get_center()
    camera_radius = np.linalg.norm(camera_center)

    sampled_cameras = sample_cameras_on_hemisphere(
        intrinsics=intrinsics,
        width=width,
        height=height,
        radius=camera_radius,
        up="z",
        nr_cameras=100,
    )

    camera = sampled_cameras[0]

    # shoot rays from camera
    rays_o, rays_d, points_2d_screen = camera.get_rays(device=device)

    plot_rays_samples(
        rays_o=rays_o.cpu().numpy(),
        rays_d=rays_d.cpu().numpy(),
        # t_near=t_near.cpu().numpy(),
        # t_far=t_far.cpu().numpy(),
        nr_rays=32,
        camera=camera,
        # bounding_boxes=[bb] if bb is not None else None,
        azimuth_deg=20,
        elevation_deg=30,
        scene_radius=camera_radius,
        # draw_bounding_cube=draw_bounding_cube,
        # draw_contraction_spheres=draw_contraction_spheres,
        title="bounding box intersections",
        show=cfg.with_viewer,
        save_path=os.path.join(output_path, "virtual_camera_rays.png"),
    )

    # Visualize cameras
    plot_3d(
        cameras=sampled_cameras,
        point_clouds=mv_data.point_clouds,
        bounding_boxes=[bb],
        azimuth_deg=20,
        elevation_deg=30,
        scene_radius=mv_data.get_scene_radius(),
        up="z",
        figsize=(15, 15),
        title="sampled cameras",
        show=cfg.with_viewer,
        save_path=os.path.join(output_path, "virtual_cameras.png"),
    )

    # Create tensor reel
    tensorreel = TensorReel(sampled_cameras, modalities=[], device=device)  # no data
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

        # cameras_idx = np.random.permutation(len(mv_data.get_split("train")))[:2]

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

            # print data shapes
            for k, v in batch.items():
                # if v is a dict
                if isinstance(v, dict):
                    for k1, v1 in v.items():
                        print(f"{k}, " f"{k1}", v1.shape, v1.device, v1.dtype)
                else:
                    print(f"{k}", v.shape, v.device, v.dtype)

            plot_current_batch(
                cameras=sampled_cameras,
                cameras_idx=batch_cameras_idx.cpu().numpy(),
                rays_o=batch_rays_o.cpu().numpy(),
                rays_d=batch_rays_d.cpu().numpy(),
                rgbs=None,
                masks=None,
                bounding_boxes=[bb],
                azimuth_deg=azimuth_deg,
                elevation_deg=30,
                scene_radius=mv_data.get_scene_radius(),
                up="z",
                figsize=(15, 15),
                title=f"rays batch sampling {i}",
                show=cfg.with_viewer,
                save_path=os.path.join(
                    output_path,
                    f"virtual_cameras_batch_{i}.png",
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