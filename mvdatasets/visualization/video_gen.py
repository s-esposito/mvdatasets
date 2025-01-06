import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Literal
from pathlib import Path
from mvdatasets.camera import Camera
from mvdatasets.visualization.matplotlib import plot_camera_trajectory
from mvdatasets.utils.printing import print_warning, print_log, print_error
from mvdatasets.geometry.primitives import BoundingBox, BoundingSphere, PointCloud


def make_video_camera_trajectory(
    cameras: list[Camera],
    save_path: Path,  # e.g. Path("./trajectory.mp4"),
    dataset_name: str = None,
    point_clouds: list[PointCloud] = None,
    nr_frames: int = -1,  # -1 means all frames
    max_nr_points: int = 10000,
    fps: int = 10,
    remove_tmp_files: bool = True,
    azimuth_deg: float = 60.0,
    elevation_deg: float = 30.0,
    scene_radius: float = 1.0,
    up: Literal["z", "y"] = "z",
    draw_origin: bool = True,
) -> None:

    # check if save_path extension is mp4
    if save_path.suffix != ".mp4":
        print_error("save_path extension must be mp4")

    # uniform sampling of sequence lenght
    sequence_len = len(cameras)
    if nr_frames == -1:
        nr_frames = sequence_len
    elif nr_frames > sequence_len or nr_frames <= 0:
        print_error(
            f"nr_frames must be less than or equal to {sequence_len} and greater than 0"
        )
    step_size = sequence_len // nr_frames
    frames_idxs = np.arange(0, sequence_len, step_size)

    # remove extension from save_path
    output_path = save_path.parent / save_path.stem

    # create output folder (e.g. ./trajectory)

    # if output_path exists, remove it
    if os.path.exists(output_path):
        print_log(f"overriding existing {output_path}")
        os.system(f"rm -rf {output_path}")
    os.makedirs(output_path)
    
    # downsample point cloud
    new_point_clouds = []
    for point_cloud in point_clouds:
        new_point_cloud = deepcopy(point_cloud)
        new_point_cloud.downsample(max_nr_points)
        new_point_clouds.append(new_point_cloud)
    point_clouds = new_point_clouds

    # Visualize cameras
    pbar = tqdm(enumerate(frames_idxs), desc="frames", ncols=100)
    for _, last_frame_idx in pbar:

        # get camera
        camera = cameras[last_frame_idx]

        # get timestamp
        ts = camera.get_timestamps()[0]
        # round to 3 decimal places
        ts = round(ts, 3)

        # save plot as png in output_path
        plot_camera_trajectory(
            cameras=cameras,
            last_frame_idx=last_frame_idx,
            draw_every_n_cameras=1,
            point_clouds=point_clouds,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            max_nr_points=None,
            up=up,
            scene_radius=scene_radius,
            draw_rgb_frame=True,
            draw_all_cameras_frames=False,
            draw_image_planes=True,
            draw_cameras_frustums=True,
            draw_origin=draw_origin,
            figsize=(15, 15),
            title=f"{dataset_name} camera trajectory up to time {ts} [s]",
            show=False,
            save_path=os.path.join(output_path, f"{format(last_frame_idx, '09d')}.png"),
        )

    # make video from plots in output_path
    os.system(
        f'ffmpeg -y -r {fps} -i {output_path}/%09d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -crf 25 -pix_fmt yuv420p {save_path}'
    )
    print_log(f"video saved at {save_path}")

    # remove tmp files
    if remove_tmp_files:
        os.system(f"rm -rf {output_path}")
        print_log("removed temporary files")
