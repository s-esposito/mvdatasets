import numpy
import PIL
import os
import sys
import time
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import imageio

from mvdatasets.utils.plotting import (
    plot_cameras,
    plot_camera_rays,
    plot_current_batch,
    plot_camera_reprojected_point_cloud,
)
from mvdatasets.utils.raycasting import get_camera_rays, get_camera_frames
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.profiler import Profiler
from torch.utils.data import DataLoader

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# # Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed(seed)  # Set a random seed for GPU
else:
    device = "cuda"
torch.set_default_device(device)

# Set default tensor type
torch.set_default_dtype(torch.float32)

# Set profiler
profiler = Profiler()  # nb: might slow down the code

datasets_path = "/home/stefano/Data"
dataset_name = "dtu"
scene_name = "dtu_scan83"

# load gt mesh if exists
gt_meshes_paths = [os.path.join("debug/meshes/", dataset_name, scene_name, "mesh.ply")]

# dataset loading
mv_data = MVDataset(
    dataset_name,
    scene_name,
    datasets_path,
    point_clouds_paths=gt_meshes_paths,
    splits=["train", "test"],
    test_camera_freq=8,
    load_mask=True,
)

# CAMERA IMAGES SCALE AND PLOTTING ----------------------------------------

# camera = deepcopy(mv_data["train"][0])
# print(camera)

# # gen rays
# rays_o, rays_d = get_camera_rays(camera)
# print("rays_o", rays_o.shape, rays_o.device)
# print("rays_d", rays_d.shape, rays_d.device)

# rgb, mask = get_camera_frames(camera)
# print("rgb", rgb.shape, rgb.device)
# print("mask", mask.shape, mask.device)

# # img = rgb.cpu().numpy().reshape(camera.height, camera.width, 3)
# # plt.imshow(img)
# # plt.show()

# # make from the gt frame a smaller frame until we reach a certain size
# while min(camera.width, camera.height) > 200:
#     print("camera.width", camera.width, "camera.height", camera.height)
#     camera.subsample(scale=0.5)
# print("camera.width", camera.width, "camera.height", camera.height)

# # gen rays
# rays_o, rays_d = get_camera_rays(camera)
# print("rays_o", rays_o.shape, rays_o.device)
# print("rays_d", rays_d.shape, rays_d.device)

# rgb, mask = get_camera_frames(camera)
# print("rgb", rgb.shape, rgb.device)
# print("mask", mask.shape, mask.device)

# # img = rgb.cpu().numpy().reshape(camera.height, camera.width, 3)
# # plt.imshow(img)
# # plt.show()

# # visualize camera
# fig = plot_camera_rays(
#     camera, 512, azimuth_deg=60, elevation_deg=30, up="y", figsize=(15, 15)
# )
# plt.show()

# exit()

# VISUALIZE ALL TRAINING CAMERAS ---------------------------------------------

# # Visualize cameras
# fig = plot_cameras(
#     mv_data["train"],
#     points=mv_data.point_clouds[0],
#     azimuth_deg=20,
#     elevation_deg=30,
#     up="y",
#     figsize=(15, 15),
# )

# plt.show()
# # plt.savefig("test_static_scenes.png", bbox_inches="tight", pad_inches=0)
# # plt.close()

# exit()

# VISUALIZE REPROJECTED POINTS ----------------------------------------------

# camera = deepcopy(mv_data["train"][23])
# print(camera)

# # make from the gt frame a smaller frame until we reach a certain size
# while min(camera.width, camera.height) > 200:
#     print("camera.width", camera.width, "camera.height", camera.height)
#     camera.subsample(scale=0.5)
# print("camera.width", camera.width, "camera.height", camera.height)

# print(camera)

# fig = plot_camera_reprojected_point_cloud(camera, mv_data.point_clouds)

# plt.show()
# # plt.savefig("test_static_scenes_projection.png", transparent=True)
# # plt.close()

# exit()

# ----------------------------------------------------------------------------

batch_size = 512

profiler.reset()

# TensorReel (~1300 it/s), camera's data in concatenated in big tensors on GPU

tensor_reel = TensorReel(mv_data["train"], device=device)

nr_iterations = 60
cameras_idx = None
frame_idx = None
pbar = tqdm(range(nr_iterations))
azimuth_deg = 0
azimuth_deg_delta = 360 / (nr_iterations / 2)
frames_paths = []
for i in pbar:
    pbar.set_description("Ray casting")

    # cameras_idx = np.random.permutation(len(mv_data["train"]))[:1]

    if profiler is not None:
        profiler.start("get_next_batch")

    # get rays and gt values
    with torch.set_grad_enabled(False):
        (
            camera_idx,
            rays_o,
            rays_d,
            gt_rgb,
            gt_mask,
            frame_idx,
        ) = tensor_reel.get_next_batch(
            batch_size=batch_size, cameras_idx=cameras_idx, frame_idx=frame_idx
        )

    if profiler is not None:
        profiler.end("get_next_batch")

    fig = plot_current_batch(
        mv_data["train"],
        camera_idx,
        rays_o,
        rays_d,
        gt_rgb,
        gt_mask,
        azimuth_deg=azimuth_deg,
        elevation_deg=30,
        up="y",
        figsize=(15, 15),
    )

    # plt.show()
    frame_path = os.path.join("plots", f"test_static_scenes_batch_{i}.png")
    plt.savefig(
        frame_path,
        # bbox_inches="tight",
        # pad_inches=0,
    )
    plt.close()
    frames_paths.append(frame_path)

    # update azimuth every 2 iterations
    if i % 2 != 0:
        azimuth_deg += azimuth_deg_delta

# # make webm video
# video_path = os.path.join("plots", "test.webm")
# writer = imageio.get_writer(video_path, fps=30, codec="libvpx-vp9")
# for frame_path in frames_paths:
#     im = imageio.imread(frame_path)
#     writer.append_data(im)
#     os.remove(frame_path)

# for i in tqdm(range(nr_iterations)):
#     profiler.start("get_next_batch")
#     idx, rays_o, rays_d, rgb, mask = get_next_batch(data_loader)
#     profiler.end("get_next_batch")

#     fig = plot_current_batch(
#         dataset_train.cameras,
#         idx,
#         rays_o,
#         rays_d,
#         rgb,
#         mask,
#         azimuth_deg=60,
#         elevation_deg=30,
#         up="y",
#         figsize=(15, 15),
#     )

#     # plt.show()
#     plt.savefig(f"test_static_scenes_batch_{i}.png", bbox_inches="tight", pad_inches=0)

if profiler is not None:
    profiler.print_avg_times()

# OLD PyTorch DataLoader (~28 it/s), camera's data in on GPU

# from mvdatasets.utils.loader import DatasetSampler, custom_collate, get_next_batch

# # Create a DataLoader for the MVDataset
# cameras_batch_size = len(dataset_train.cameras)  # alway sample from all cameras
# max_rays_batch_size = batch_size
# per_camera_rays_batch_size = max_rays_batch_size // cameras_batch_size
# dataset_train.per_camera_rays_batch_size = per_camera_rays_batch_size
# # nr_workers = 1
# custom_sampler = DatasetSampler(dataset_train, shuffle=True)
# data_loader = DataLoader(
#     dataset=dataset_train,
#     batch_size=cameras_batch_size,
#     sampler=custom_sampler,
#     collate_fn=custom_collate,
#     # num_workers=nr_workers,
# )

# nr_iterations = 1000
# pbar = tqdm(range(nr_iterations))
# for i in pbar:
#     pbar.set_description("Ray casting")

#     if profiler is not None:
#         profiler.start("get_next_batch")

#     # get rays and gt values
#     with torch.set_grad_enabled(False):
#         camera_idx, rays_o, rays_d, gt_rgb, gt_mask, frame_idx = get_next_batch(
#             data_loader
#         )

#         if device != "cpu":
#             camera_idx = camera_idx.to(device)
#             rays_o = rays_o.to(device)
#             rays_d = rays_d.to(device)
#             gt_rgb = gt_rgb.to(device)
#             gt_mask = gt_mask.to(device)
#             frame_idx = frame_idx.to(device)

#         # print("camera_idx", camera_idx.shape, camera_idx.device)
#         # print("rays_o", rays_o.shape, rays_o.device)
#         # print("rays_d", rays_d.shape, rays_d.device)
#         # print("gt_rgb", gt_rgb.shape, gt_rgb.device)
#         # print("gt_mask", gt_mask.shape, gt_mask.device)
#         # print("frame_idx", frame_idx.shape, frame_idx.device)

#     if profiler is not None:
#         profiler.end("get_next_batch")
