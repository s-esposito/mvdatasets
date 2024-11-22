import tyro
import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from config import get_dataset_test_preset
from config import Args
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.visualization.matplotlib import plot_image


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

    # random camera index
    rand_idx = 0  # torch.randint(0, len(mv_data["test"]), (1,)).item()
    camera = deepcopy(mv_data["test"][rand_idx])
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
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_bounding_box.png"),
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
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_bounding_sphere.png"),
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
