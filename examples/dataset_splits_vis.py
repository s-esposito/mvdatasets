import tyro
import numpy as np
import os
import sys
from pathlib import Path
from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.utils.printing import print_error, print_warning
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

    # # sdf init
    # bs = BoundingSphere(
    #     pose=np.eye(4),
    #     local_scale=mv_data.get_sphere_init_radius(),
    #     device=device,
    #     verbose=True,
    # )

    # foreground bb
    bb = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.get_foreground_radius() * 2,
        device=device,
    )

    # scene type
    scene_type = mv_data.get_scene_type()
    if scene_type == "bounded":
        draw_bounding_cube = True
        draw_contraction_spheres = False

    if scene_type == "unbounded":
        draw_bounding_cube = False
        draw_contraction_spheres = True

        if mv_data.get_scene_radius() > 1.0:
            print_warning(
                "scene radius is greater than 1.0, contraction spheres will not be displayed"
            )
            bb = None
            draw_contraction_spheres = False

    # Visualize cameras
    for split in mv_data.get_splits():

        print("visualizing cameras for split: ", split)

        nr_cameras = len(mv_data.get_split(split))
        if nr_cameras // 50 > 1:
            draw_every_n_cameras = nr_cameras // 50
            print_warning(
                f"{split} has too many cameras; displaying one every {draw_every_n_cameras}"
            )
        else:
            draw_every_n_cameras = 1

        plot_3d(
            cameras=mv_data.get_split(split),
            draw_every_n_cameras=draw_every_n_cameras,
            point_clouds=mv_data.point_clouds,
            bounding_boxes=[bb] if bb is not None else [],
            # bounding_spheres=[bs],
            azimuth_deg=20,
            elevation_deg=30,
            up="z",
            scene_radius=mv_data.get_scene_radius(),
            draw_bounding_cube=draw_bounding_cube,
            draw_image_planes=True,
            draw_cameras_frustums=False,
            draw_contraction_spheres=draw_contraction_spheres,
            figsize=(15, 15),
            title=f"{split} cameras",
            show=cfg.with_viewer,
            save_path=os.path.join(
                output_path, f"{dataset_name}_{scene_name}_{split}_cameras.png"
            ),
        )

    print("done")


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