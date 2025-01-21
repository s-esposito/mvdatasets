import tyro
import os
import sys
import torch
from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.visualization.matplotlib import plot_image
from mvdatasets.configs.example_config import ExampleConfig
from examples import get_dataset_test_preset, custom_exception_handler
from mvdatasets.utils.printing import print_warning


def main(cfg: ExampleConfig, pc_paths: list[Path]):

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
        config=cfg.data,
        point_clouds_paths=pc_paths,
        verbose=True,
    )

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data.get_split("test")), (1,)).item()
    camera = deepcopy(mv_data.get_split("test")[rand_idx])
    # shoot rays from camera
    rays_o, rays_d, points_2d_screen = camera.get_rays(device=device)

    # bounding box
    bounding_volume = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.get_foreground_radius() * 2,
        device=device,
    )
    # bounding_volume intersection test
    is_hit, t_near, t_far, p_near, p_far = bounding_volume.intersect(rays_o, rays_d)
    hit_range = t_far - t_near
    hit_range = hit_range.cpu().numpy()
    # get the color map
    color_map = plt.colormaps.get_cmap("jet")
    # apply the colormap
    hit_range = color_map(hit_range)[:, :3]

    data = camera.get_data(keys=["rgbs"])
    rgbs = data["rgbs"].cpu().numpy()
    img_np = (rgbs * 0.5) + (hit_range * 0.5)
    img_np = img_np.reshape(camera.width, camera.height, -1)

    plot_image(
        image=img_np,
        title="Bounding Box",
        show=cfg.with_viewer,
        save_path=os.path.join(
            output_path, f"{dataset_name}_{scene_name}_bounding_box.png"
        ),
    )

    # bounding sphere
    bounding_volume = BoundingSphere(
        pose=np.eye(4), local_scale=mv_data.get_foreground_radius(), device=device
    )
    # bounding_volume intersection test
    is_hit, t_near, t_far, p_near, p_far = bounding_volume.intersect(rays_o, rays_d)
    hit_range = t_far - t_near
    hit_range = hit_range.cpu().numpy()
    # get the color map
    color_map = plt.colormaps.get_cmap("jet")
    # apply the colormap
    hit_range = color_map(hit_range)[:, :3]

    data = camera.get_data(keys=["rgbs"])
    rgbs = data["rgbs"].cpu().numpy()
    img_np = (rgbs * 0.5) + (hit_range * 0.5)
    img_np = img_np.reshape(camera.width, camera.height, -1)

    plot_image(
        image=img_np,
        title="Bounding Sphere",
        show=cfg.with_viewer,
        save_path=os.path.join(
            output_path, f"{dataset_name}_{scene_name}_bounding_sphere.png"
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
