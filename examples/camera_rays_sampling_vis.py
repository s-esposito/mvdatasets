import tyro
import os
import sys
import torch
from pathlib import Path
import numpy as np
from copy import deepcopy
from typing import List
from mvdatasets.visualization.matplotlib import plot_3d, plot_rays_samples
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives import BoundingBox, BoundingSphere
from mvdatasets.utils.printing import print_error, print_warning
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

    # list of bounding boxes to draw
    bb = None
    bs = None

    scene_type = mv_data.get_scene_type()
    if scene_type == "bounded":
        draw_bounding_cube = True
        draw_contraction_spheres = False
        # bounding box
        bb = BoundingBox(
            pose=np.eye(4),
            local_scale=mv_data.get_foreground_radius() * 2,
            device=device,
        )
    elif scene_type == "unbounded":
        draw_bounding_cube = False
        draw_contraction_spheres = True
        if mv_data.get_scene_radius() > 1.0:
            print_warning(
                "scene radius is greater than 1.0, contraction spheres will not be displayed"
            )
            draw_contraction_spheres = False

    else:
        raise ValueError("scene_type not supported")

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data.get_split("test")), (1,)).item()
    camera = deepcopy(mv_data.get_split("train")[rand_idx])

    # resize camera
    camera.resize(subsample_factor=10)

    # Visualize cameras
    plot_3d(
        cameras=[camera],
        point_clouds=mv_data.point_clouds,
        # bounding_boxes=[bb] if bb is not None else None,
        nr_rays=256,
        azimuth_deg=20,
        elevation_deg=30,
        scene_radius=mv_data.get_scene_radius(),
        draw_bounding_cube=draw_bounding_cube,
        up="z",
        draw_image_planes=True,
        draw_cameras_frustums=False,
        draw_contraction_spheres=draw_contraction_spheres,
        figsize=(15, 15),
        title=f"test camera {rand_idx} rays",
        show=cfg.with_viewer,
        save_path=os.path.join(
            output_path, f"{dataset_name}_{scene_name}_camera_rays.png"
        ),
    )

    # Visualize intersections with bounding box

    # shoot rays from camera
    rays_o, rays_d, points_2d_screen = camera.get_rays(device=device)

    # bounding primitive intersection test
    if scene_type == "bounded":
        is_hit, t_near, t_far, p_near, p_far = bb.intersect(rays_o, rays_d)
    elif scene_type == "unbounded":
        # bounding sphere
        bs = BoundingSphere(
            pose=np.eye(4),
            local_scale=0.5,
            device=device,
        )
        is_hit, t_near, t_far, p_near, p_far = bs.intersect(rays_o, rays_d)
        if mv_data.get_scene_radius() > 1.0:
            print_warning(
                "scene radius is greater than 1.0, bounding box is not defined"
            )
            exit(0)
    else:
        raise ValueError("scene_type not supported")

    plot_rays_samples(
        rays_o=rays_o.cpu().numpy(),
        rays_d=rays_d.cpu().numpy(),
        t_near=t_near.cpu().numpy(),
        t_far=t_far.cpu().numpy(),
        nr_rays=32,
        camera=camera,
        bounding_boxes=[bb] if bb is not None else None,
        azimuth_deg=20,
        elevation_deg=30,
        scene_radius=mv_data.get_scene_radius(),
        draw_bounding_cube=draw_bounding_cube,
        draw_contraction_spheres=draw_contraction_spheres,
        title="bounding box intersections",
        show=False,
        save_path=os.path.join(
            output_path, f"{dataset_name}_{scene_name}_bounding_box_intersections.png"
        ),
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
