import numpy
import PIL
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm

from datasets.utils.plotting import plot_cameras, plot_camera_rays, plot_current_batch
from datasets.mv_dataset import MVDataset
from datasets.utils.images import tensor2image
from datasets.utils.geometry import project_points_3d_to_2d

from torch.utils.data import DataLoader

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(seed)  # Set a random seed for GPU
else:
    device = "cpu"
torch.set_default_device(device)

# Set default tensor type
torch.set_default_dtype(torch.float32)

data_path = "/home/stefano/Data"
dataset_name = "dtu"
scene_name = "dtu_scan83"

scene_data_path = os.path.join(data_path, dataset_name, scene_name)
# make sure folder exists
assert os.path.exists(scene_data_path), "Scene data path does not exist"
print("Scene data path: {}".format(scene_data_path))

# load gt mesh if exists
gt_meshes_paths = [os.path.join("debug/meshes/", dataset_name, scene_name, "mesh.ply")]

# dataset loading
dataset_train = MVDataset(
    dataset_name,
    scene_name,
    scene_data_path,
    point_clouds_paths=gt_meshes_paths,
    split="all",
    use_every_for_test_split=8,
    auto_center_method="none",  # "poses", "focus", "none"
    auto_orient_method="none",  # "up", "none"
    # auto_scale_poses=False,
    device=device,
)

# T = dataset_train.cameras[0].transform
# K = dataset_train.cameras[0].intrinsics
# Rt = dataset_train.cameras[0].get_pose()
# P = dataset_train.cameras[0].get_projection()

# print("T", T)
# print("K", K)
# print("Rt", Rt)
# print("P", P)

# fig = plot_cameras(
#     dataset_train.cameras,
#     points=dataset_train.point_clouds[0],
#     azimuth_deg=20,
#     elevation_deg=30,
#     up="y",
#     figsize=(15, 15),
# )

# plt.show()
# plt.savefig("test_static_scenes.png", bbox_inches="tight", pad_inches=0)

# img_torch = dataset_train.cameras[0].imgs[0]
# print("img_torch", img_torch.shape)
# img_pil = tensor2image(img_torch)
# img_pil.save("test_static_scenes_img.png")

# img_torch = dataset_train.cameras[0].masks[0]
# print("img_torch", img_torch.shape)
# img_pil = tensor2image(img_torch)
# img_pil.save("test_static_scenes_mask.png")

# camera_idx = 3
# img_np = dataset_train.cameras[camera_idx].get_frame().cpu().numpy()
# intrinsics = dataset_train.cameras[camera_idx].intrinsics.cpu().numpy()
# points_2d = project_points_3d_to_2d(
#     points_3d=dataset_train.point_clouds[0],
#     intrinsics=intrinsics,
#     c2w=dataset_train.cameras[camera_idx].get_pose().cpu().numpy(),
# )
# # filter out points outside image range
# points_2d = points_2d[points_2d[:, 0] > 0]
# points_2d = points_2d[points_2d[:, 1] > 0]
# points_2d = points_2d[points_2d[:, 0] < img_np.shape[1]]
# points_2d = points_2d[points_2d[:, 1] < img_np.shape[0]]
# print("points_2d", points_2d.shape)

# fig = plt.figure()
# plt.imshow(img_np, alpha=1.0)
# colors = np.column_stack([points_2d, np.zeros((points_2d.shape[0], 1))])
# colors /= np.max(colors)
# colors += 0.5
# colors /= np.max(colors)
# plt.scatter(points_2d[:, 0], points_2d[:, 1], s=5, c=colors, marker=".")
# plt.gca().set_aspect("equal", adjustable="box")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig("test_static_scenes_projection.png", transparent=True)

# camera = dataset_train.cameras[0]

# fig = plot_camera_rays(
#     camera, 512, azimuth_deg=60, elevation_deg=30, up="y", figsize=(15, 15)
# )

# plt.show()

# Create a DataLoader for the MVDataset
cameras_batch_size = len(dataset_train.cameras)  # Set your desired batch size
rays_batch_size = 512
per_camera_rays_batch_size = rays_batch_size // cameras_batch_size
dataset_train.per_camera_rays_batch_size = per_camera_rays_batch_size
# print("cameras_batch_size", cameras_batch_size)
# print("rays_batch_size", rays_batch_size)
# print("per_camera_rays_batch_size", per_camera_rays_batch_size)

# # Inside the training loop or wherever you create the DataLoader
# if torch.cuda.is_available():
#     data_loader = DataLoader(
#         dataset=dataset_train,
#         batch_size=cameras_batch_size,
#         shuffle=True,
#         pin_memory=True,
#     )
# else:
#     data_loader = DataLoader(
#         dataset=dataset_train, batch_size=cameras_batch_size, shuffle=True
#     )

from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.nr_samples = len(dataset.cameras)
        self.seed = 42

    def __iter__(self):
        return iter(torch.randperm(self.nr_samples))

    def __len__(self):
        return self.nr_samples


def custom_collate(batch):
    idx, rays_o, rays_d, rgb, mask = zip(*batch)
    return (
        torch.stack(idx),
        torch.stack(rays_o),
        torch.stack(rays_d),
        torch.stack(rgb),
        torch.stack(mask),
    )


custom_sampler = CustomSampler(dataset_train)

data_loader = DataLoader(
    dataset=dataset_train,
    batch_size=cameras_batch_size,
    sampler=custom_sampler,
    collate_fn=custom_collate,
)


def get_next_batch(data_loader):
    """calls next on the data_loader and returns a batch of data"""
    idx, rays_o, rays_d, rgb, mask = next(iter(data_loader))
    idx = idx.repeat_interleave(per_camera_rays_batch_size)
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    rgb = rgb.view(-1, 3)
    mask = mask.view(-1, 1)

    return idx, rays_o, rays_d, rgb, mask


nr_iterations = 10
for i in tqdm(range(nr_iterations)):
    idx, rays_o, rays_d, rgb, mask = get_next_batch(data_loader)

    fig = plot_current_batch(
        dataset_train.cameras,
        idx,
        rays_o,
        rays_d,
        rgb,
        mask,
        azimuth_deg=60,
        elevation_deg=30,
        up="y",
        figsize=(15, 15),
    )

    # plt.show()
    plt.savefig(
        f"test_static_scenes_batch_{i}.png", bbox_inches="tight", pad_inches=0
    )
