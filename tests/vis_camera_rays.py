import tyro
import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from config import get_dataset_test_preset
from config import Args
from mvdatasets.visualization.matplotlib import plot_3d, plot_image, plot_rays_samples
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives import BoundingBox


def main(args: Args):

    device = args.device
    dataset_path = args.datasets_path
    dataset_name = args.dataset_name
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        dataset_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True,
    )

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
        nr_rays=256,
        azimuth_deg=20,
        elevation_deg=30,
        scene_radius=mv_data.scene_radius,
        up="z",
        draw_image_planes=True,
        draw_cameras_frustums=False,
        figsize=(15, 15),
        title=f"test camera {rand_idx} rays",
        show=True,
        save_path=os.path.join("plots", f"{dataset_name}_camera_rays.png"),
    )
    
    # Visualize intersections with bounding box
    
    # bounding box
    bounding_volume = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.scene_radius*2,
        device=device,
    )
    
    # shoot rays from camera
    rays_o, rays_d, points_2d_screen = camera.get_rays(device=device)
    
    # bounding_volume intersection test
    is_hit, t_near, t_far, p_near, p_far = bounding_volume.intersect(rays_o, rays_d)

    plot_rays_samples(
        rays_o=rays_o.cpu().numpy(),
        rays_d=rays_d.cpu().numpy(),
        t_near=t_near.cpu().numpy(),
        t_far=t_far.cpu().numpy(),
        nr_rays=32,
        camera=camera,
        bounding_boxes=[bounding_volume],
        scene_radius=mv_data.scene_radius,
        title="bounding box intersections",
        show=True,
        save_path=os.path.join("plots", f"{dataset_name}_bounding_box_intersections.png"),
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)