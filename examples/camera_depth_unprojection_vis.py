import tyro
from rich import print
import os
import sys
from pathlib import Path
from typing import List
from copy import deepcopy
import matplotlib.pyplot as plt
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.visualization.matplotlib import plot_camera_2d, plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.printing import print_error, print_warning
from mvdatasets.visualization.video_gen import make_video_depth_unproject
from mvdatasets.configs.example_config import ExampleConfig
from examples import get_dataset_test_preset, custom_exception_handler


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

    #
    split_modalities = mv_data.get_split_modalities("train")
    print("split_modalities", split_modalities)
    # make sure mv_data has depth modality
    if "depths" not in split_modalities:
        raise ValueError("Dataset does not have depth modality")

    # check if camera trajectory is available
    print("nr_sequence_frames:", mv_data.get_nr_sequence_frames())
    if mv_data.get_nr_sequence_frames() <= 1:
        raise ValueError(
            f"{dataset_name} is a static dataset and does not have camera trajectory"
        )
        return

    # check if monocular sequence
    print("nr_per_camera_frames:", mv_data.get_nr_per_camera_frames())
    if mv_data.get_nr_per_camera_frames() > 1:
        raise ValueError(f"{dataset_name} is not a monocular sequence")
        return

    from mvdatasets.utils.raycasting import get_points_2d_screen_from_pixels

    # iterate over training cameras
    pcs = []
    for camera in mv_data.get_split("train"):
        # get rgb and depth images
        depth = camera.get_depth()  # (H, W, 1)
        # invert H and W
        depth = depth.transpose(1, 0, 2)  # (W, H, 1)
        # flatten depth image
        depth = depth.flatten()  # (H*W,)
        #
        zero_depth_mask = depth < 1e-3
        # get rgb
        rgb = camera.get_rgb()  # (H, W, 3)
        # invert H and W
        rgb = rgb.transpose(1, 0, 2)  # (W, H, 3)
        points_rgb = rgb.reshape(-1, 3)  # (H*W, 3)
        # get pixel coordinates
        pixels = camera.get_pixels()  # (W, H, 2)
        # get pixels centers
        points_2d_screen = get_points_2d_screen_from_pixels(pixels)  # (H*W, 2)
        # filtering
        points_2d_screen = points_2d_screen[~zero_depth_mask]
        depth = depth[~zero_depth_mask]
        points_rgb = points_rgb[~zero_depth_mask]
        # unproject depth to 3D
        points_3d = camera.unproject_points_2d_screen_to_3d_world(
            points_2d_screen=points_2d_screen, depth=depth
        )
        # create point cloud
        pc = PointCloud(
            points_3d=points_3d,
            points_rgb=points_rgb,
        )
        # append
        pcs.append(pc)

    # # create mask for filtering points
    # max_nr_points = 10000
    # if max_nr_points >= points_3d.shape[0]:
    #     # no need to filter
    #     pass
    # else:
    #     idxs = np.random.choice(points_3d.shape[0], max_nr_points, replace=False)
    #     for pc in pcs:
    #         pc.mask(idxs)

    # make video
    make_video_depth_unproject(
        cameras=mv_data.get_split("train"),
        point_clouds=pcs,
        dataset_name=dataset_name,
        remove_tmp_files=True,
        scene_radius=mv_data.get_scene_radius(),
        azimuth_deg=280.0,
        elevation_deg=5.0,
        save_path=Path(
            os.path.join(
                output_path, f"{dataset_name}_{scene_name}_depth_unproject.mp4"
            )
        ),
        fps=10,
    )

    # # plot point clouds and camera
    # plot_3d(
    #     cameras=mv_data.get_split("train"),
    #     point_clouds=pcs,
    #     azimuth_deg=20,
    #     elevation_deg=30,
    #     up="z",
    #     scene_radius=mv_data.get_scene_radius(),
    #     draw_bounding_cube=True,
    #     draw_image_planes=True,
    #     figsize=(15, 15),
    #     title="point cloud unprojection",
    #     show=cfg.with_viewer,
    #     # save_path=os.path.join(output_path, f"{dataset_name}_{scene_name}_point_cloud_from_depths.png"),
    # )


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
