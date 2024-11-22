import tyro
import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from config import get_dataset_test_preset
from config import Args
from mvdatasets.visualization.matplotlib import plot_3d, plot_rays_samples
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives import BoundingBox, BoundingSphere
from mvdatasets.utils.printing import print_error


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True,
    )
    
    bbs = []
    draw_bounding_cube = True
    draw_contraction_spheres = False
    scene_type = config.get("scene_type", None)
    if scene_type == "bounded":
        # bounding box
        bb = BoundingBox(
            pose=np.eye(4),
            local_scale=mv_data.get_foreground_radius() * 2,
            device=device,
        )
        bbs.append(bb)
    elif scene_type == "unbounded":
        draw_bounding_cube = False
        draw_contraction_spheres = True
        # bounding sphere
        bs = BoundingSphere(
            pose=np.eye(4),
            local_scale=0.5,
            device=device,
        )

    else:
        print_error("scene_type not supported")

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        point_cloud = None

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])

    # resize camera
    camera.resize(subsample_factor=10)

    # Visualize cameras
    plot_3d(
        cameras=[camera],
        points_3d=[point_cloud],
        bounding_boxes=bbs,
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
        show=True,
        save_path=os.path.join("plots", f"{dataset_name}_camera_rays.png"),
    )
    
    # Visualize intersections with bounding box
    
    # shoot rays from camera
    rays_o, rays_d, points_2d_screen = camera.get_rays(device=device)
    
    # bounding primitive intersection test
    if scene_type == "bounded":
        is_hit, t_near, t_far, p_near, p_far = bb.intersect(rays_o, rays_d)
    elif scene_type == "unbounded":
        is_hit, t_near, t_far, p_near, p_far = bs.intersect(rays_o, rays_d)
    else:
        print_error("scene_type not supported")

    plot_rays_samples(
        rays_o=rays_o.cpu().numpy(),
        rays_d=rays_d.cpu().numpy(),
        t_near=t_near.cpu().numpy(),
        t_far=t_far.cpu().numpy(),
        nr_rays=32,
        camera=camera,
        bounding_boxes=bbs,
        scene_radius=mv_data.get_scene_radius(),
        draw_bounding_cube=draw_bounding_cube,
        draw_contraction_spheres=draw_contraction_spheres,
        title="bounding box intersections",
        show=True,
        save_path=os.path.join("plots", f"{dataset_name}_bounding_box_intersections.png"),
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)