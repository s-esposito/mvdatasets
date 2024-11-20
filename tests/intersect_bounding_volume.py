import os
import sys
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from config import get_dataset_test_preset
from config import DATASETS_PATH, DEVICE, SEED

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# library imports
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere


def main(dataset_name, device):

    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        DATASETS_PATH,
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
    print("rays_o", rays_o.shape)
    print("rays_d", rays_d.shape)
    
    # bounding box
    scene_radius = mv_data.scene_radius
    scene_diameter = scene_radius * 2
    bounding_volume = BoundingBox(
        pose=np.eye(4),
        local_scale=np.array([scene_diameter, scene_diameter, scene_diameter]),
        device=device,
    )
    # bounding_volume intersection test
    is_hit, t_near, t_far, p_near, p_far = bounding_volume.intersect(rays_o, rays_d)
    hit_range = (t_far - t_near).reshape(camera.height, camera.width)
    hit_range = hit_range.cpu().numpy()
    # get the color map
    color_map = plt.colormaps.get_cmap("jet")
    # apply the colormap
    hit_range = color_map(hit_range)[:, :, :3]
    # print("hit range", hit_range)

    img_np = ((camera.get_rgb() / 255.0) * 0.5) + (hit_range * 0.5)
    plt.imshow(img_np)

    plt.savefig(
        os.path.join("plots", f"{dataset_name}_bounding_box.png"),
        transparent=True,
        dpi=300,
    )
    plt.close()

    # bounding sphere
    bounding_volume = BoundingSphere(
        pose=np.eye(4), local_scale=scene_radius, device=device
    )
    # bounding_volume intersection test
    is_hit, t_near, t_far, p_near, p_far = bounding_volume.intersect(rays_o, rays_d)
    hit_range = (t_far - t_near).reshape(camera.height, camera.width)
    hit_range = hit_range.cpu().numpy()
    # get the color map
    color_map = plt.colormaps.get_cmap("jet")
    # apply the colormap
    hit_range = color_map(hit_range)[:, :, :3]
    # print("hit range", hit_range)

    img_np = ((camera.get_rgb() / 255.0) * 0.5) + (hit_range * 0.5)
    plt.imshow(img_np)
    plt.savefig(
        os.path.join("plots", f"{dataset_name}_bounding_sphere.png"),
        transparent=True,
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":

    # Set a random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)  # Set a random seed for GPU
    torch.set_default_device(DEVICE)

    # Set default tensor type
    torch.set_default_dtype(torch.float32)

    # Get dataset test preset

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "dtu"

    main(dataset_name, DEVICE)
