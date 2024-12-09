import tyro
import numpy as np
import os
from tqdm import tqdm
from config import Args
from config import get_dataset_test_preset
from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.visualization.matplotlib import plot_current_batch
from mvdatasets.visualization.matplotlib import plot_rays_samples
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.virtual_cameras import sample_cameras_on_hemisphere
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.utils.printing import print_warning, print_error


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    test_preset = get_dataset_test_preset(dataset_name)
    scene_name = test_preset["scene_name"]
    pc_paths = test_preset["pc_paths"]
    splits = test_preset["splits"]

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=splits,
        verbose=True,
    )

    bb = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.get_foreground_radius() * 2,
        device=device,
    )

    # only available for object centric datasets
    if not mv_data.cameras_on_hemisphere:
        print_error(f"{dataset_name} does not have cameras on hemisphere")

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
        show=True,
        save_path=os.path.join("plots", "sampled_camera.png"),
    )

    exit(0)

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
        show=True,
        save_path=os.path.join("plots", f"{dataset_name}_sampled_cameras.png"),
    )

    exit(0)

    # Create tensor reel
    tensor_reel = TensorReel(sampled_cameras, modalities=[], device=device)  # no data

    nr_cameras = len(mv_data.get_split("train"))
    nr_per_camera_frames = mv_data.get_nr_per_camera_frames()
    benchmark = False
    batch_size = 512
    nr_iterations = 10
    cameras_idx = np.random.permutation(nr_cameras)[:5]
    frames_idx = np.random.permutation(nr_per_camera_frames)[:2]
    print("cameras_idx", cameras_idx)
    print("frames_idx", frames_idx)
    pbar = tqdm(range(nr_iterations), desc="ray casting", ncols=100)
    azimuth_deg = 20
    azimuth_deg_delta = 1  # 360 / (nr_iterations / 2)
    for i in pbar:

        # cameras_idx = np.random.permutation(len(mv_data.get_split("train")))[:2]

        if profiler is not None:
            profiler.start("get_next_rays_batch")

        # get rays and gt values
        batch = tensor_reel.get_next_rays_batch(
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
            batch_frames_idx = batch["frames_idx"]
            batch_timestamps = batch["timestamps"]

            # print data shapes
            print("batch_cameras_idx", batch_cameras_idx.shape, batch_cameras_idx.dtype)
            print("batch_frames_idx", batch_frames_idx.shape, batch_frames_idx.dtype)
            print(
                "batch_rays_o",
                batch_rays_o.shape,
                batch_rays_o.device,
                batch_rays_o.dtype,
            )
            print(
                "batch_rays_d",
                batch_rays_d.shape,
                batch_rays_d.device,
                batch_rays_d.dtype,
            )
            print(
                "batch_timestamps",
                batch_timestamps.shape,
                batch_timestamps.device,
                batch_timestamps.dtype,
            )
            for k, v in batch_vals.items():
                if v is not None:
                    print(f"{k}", v.shape, v.device, v.dtype)

            # print("timestamps", batch_timestamps)

            plot_current_batch(
                cameras=sampled_cameras,
                cameras_idx=batch_cameras_idx,
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
                show=False,
                save_path=os.path.join(
                    "plots", f"{dataset_name}_sampled_cameras_batch_{i}.png"
                ),
            )

            # update azimuth
            azimuth_deg += azimuth_deg_delta

    if profiler is not None:
        profiler.print_avg_times()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
