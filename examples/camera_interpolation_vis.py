import tyro
import numpy as np
import os
import sys
from typing import List
from pathlib import Path
from mvdatasets.visualization.matplotlib import plot_camera_trajectory
from mvdatasets.visualization.video_gen import make_video_camera_trajectory
from mvdatasets.mvdataset import MVDataset
from mvdatasets import Camera
from mvdatasets.utils.printing import print_warning
from mvdatasets.configs.example_config import ExampleConfig
from examples import get_dataset_test_preset, custom_exception_handler
from mvdatasets.geometry.trajectories import (
    generate_spiral_path,
    generate_ellipse_path_z,
    generate_ellipse_path_y,
    generate_interpolated_path,
)


def main(cfg: ExampleConfig, pc_paths: List[Path]):

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
        config=cfg.data.asdict(),
        point_clouds_paths=pc_paths,
        verbose=True,
    )

    # get all camera poses in split
    cameras = mv_data.get_split("train")[1:3]
    # collect all camera poses
    c2w_all = [camera.get_pose() for camera in cameras]
    # concatenate all camera poses
    c2w_all = np.stack(c2w_all, axis=0)
    # interpolate camera poses
    c2w_all = generate_interpolated_path(cameras=cameras, n_interp=100)

    # use camera 0 as a template to create all new cameras
    camera = cameras[0]
    new_cameras = []
    for i, c2w in enumerate(c2w_all):
        new_camera = Camera(
            intrinsics=camera.get_intrinsics(),
            pose=c2w,
            camera_label=str(i),
            width=camera.get_width(),
            height=camera.get_height(),
            near=camera.get_near(),
            far=camera.get_far(),
            temporal_dim=1,
            subsample_factor=1,
        )
        new_cameras.append(new_camera)

    # make video
    make_video_camera_trajectory(
        cameras=new_cameras,
        point_clouds=mv_data.point_clouds,
        dataset_name=dataset_name,
        nr_frames=-1,  # -1 means all frames
        remove_tmp_files=True,
        scene_radius=mv_data.get_scene_radius(),
        save_path=Path(
            os.path.join(output_path, f"{dataset_name}_{scene_name}_interpolation.mp4")
        ),
        fps=10,
    )


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
        print_warning(
            f"scene_name is None, using preset test scene {args.scene_name} for dataset"
        )
    # additional point clouds paths (if any)
    pc_paths = test_preset["pc_paths"]

    # start the example program
    main(args, pc_paths)


# elif traj_path == "ellipse":
#     height = c2w_all[:, 2, 3].mean()
#     c2w_all = generate_ellipse_path_z(c2w_all, height=height)

# # elif traj_path == "spiral":
# #     c2w_all = generate_spiral_path(
# #         c2w_all,
# #         bounds=bounds * scene_radius,
# #         spiral_scale_r=spiral_radius_scale,
# #     )
