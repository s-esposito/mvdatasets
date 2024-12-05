import tyro
import os
import numpy as np
from pathlib import Path
from config import get_dataset_test_preset
from config import Args
from mvdatasets.visualization.matplotlib import plot_camera_trajectory
from mvdatasets.visualization.video_gen import make_video_camera_trajectory
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.printing import print_warning, print_log


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    test_preset = get_dataset_test_preset(dataset_name)
    scene_name = test_preset["scene_name"]
    pc_paths = test_preset["pc_paths"]
    config = test_preset["config"]
    splits = test_preset["splits"]

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=splits,
        config=config,
        verbose=True,
    )

    # check if camera trajectory is available
    print("nr_sequence_frames:", mv_data.get_nr_sequence_frames())
    if mv_data.get_nr_sequence_frames() <= 1:
        print_warning(
            f"{dataset_name} is a static dataset and does not have camera trajectory"
        )
        return

    # check if monocular sequence
    print("nr_per_camera_frames:", mv_data.get_nr_per_camera_frames())
    if mv_data.get_nr_per_camera_frames() > 1:
        print_warning(f"{dataset_name} is not a monocular sequence")
        return

    # make video
    make_video_camera_trajectory(
        cameras=mv_data.get_split("train"),
        points_3d=mv_data.point_clouds,
        dataset_name=dataset_name,
        nr_frames=-1,  # -1 means all frames
        remove_tmp_files=True,
        scene_radius=mv_data.get_scene_radius(),
        save_path=Path(os.path.join("plots", f"{dataset_name}_trajectory.mp4")),
        fps=10,
    )

    # # create output folder
    # output_path = os.path.join("plots", f"{dataset_name}_trajectory")
    # os.makedirs(output_path, exist_ok=True)

    # # Visualize cameras
    # for _, last_frame_idx in enumerate(frames_idxs):

    #     # get camera
    #     camera = mv_data.get_split("train")[last_frame_idx]

    #     # get timestamp
    #     ts = camera.get_timestamps()[0]
    #     # round to 3 decimal places
    #     ts = round(ts, 3)

    #     plot_camera_trajectory(
    #         cameras=mv_data.get_split("train"),
    #         last_frame_idx=last_frame_idx,
    #         draw_every_n_cameras=1,
    #         points_3d=[point_cloud],
    #         points_3d_colors=["black"],
    #         azimuth_deg=60,
    #         elevation_deg=30,
    #         max_nr_points=None,
    #         up="z",
    #         scene_radius=mv_data.get_scene_radius(),
    #         draw_rgb_frame=True,
    #         draw_all_cameras_frames=False,
    #         draw_image_planes=True,
    #         draw_cameras_frustums=True,
    #         figsize=(15, 15),
    #         title=f"{dataset_name} camera trajectory up to time {ts} [s]",
    #         show=False,
    #         save_path=os.path.join(output_path, f"{dataset_name}_trajectory_{format(last_frame_idx, '09d')}.png"),
    #     )

    # # make video
    # video_path = os.path.join(output_path, f"{dataset_name}_trajectory.mp4")
    # os.system(f'ffmpeg -y -r 10 -i {output_path}/{dataset_name}_trajectory_%09d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -crf 25 -pix_fmt yuv420p {video_path}')
    # print_log(f"Video saved at {video_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
