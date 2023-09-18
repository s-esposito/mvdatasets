import os
from glob import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from datasets.utils.camera import Camera


def load_dtu(
    data_path,
    load_with_mask=True,
    # downscale_factor=1
):
    # create cameras objects
    cameras = []

    # load images to cpu as numpy arrays
    imgs = []
    images_list = sorted(glob(os.path.join(data_path, "image/*.png")))
    pbar = tqdm(images_list, desc="Loading images", leave=True)
    for im_name in pbar:
        # load PIL image
        img = Image.open(im_name)
        img = np.array(img) / 255.0
        img = torch.from_numpy(img)
        # img = img[::downscale_factor, ::downscale_factor]
        imgs.append(img)

    # (optional) load mask images to cpu as numpy arrays
    masks = []
    if load_with_mask:
        masks_list = sorted(glob(os.path.join(data_path, "mask/*.png")))
        pbar = tqdm(masks_list, desc="Loading masks", leave=True)
        for im_name in pbar:
            # load PIL image
            mask = Image.open(im_name)
            mask = np.array(mask)[:, :, 0, None] / 255.0
            mask = torch.from_numpy(mask)
            # mask = mask[::downscale_factor, ::downscale_factor]
            masks.append(mask)

    camera_dict = np.load(os.path.join(data_path, "cameras_sphere.npz"))
    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict[f"world_mat_{idx}"] for idx in range(len(images_list))]
    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [
        camera_dict["scale_mat_%d" % idx] for idx in range(len(images_list))
    ]

    # decompose into intrinsics and extrinsics
    for idx, mats in enumerate(zip(world_mats_np, scale_mats_np)):
        world_mat_np, scale_mat_np = mats
        c2w = (world_mat_np @ scale_mat_np)[:3]
        c2w = torch.from_numpy(c2w)
        if load_with_mask:
            camera = Camera(c2w=c2w, img=imgs[idx], mask=masks[idx])
        else:
            camera = Camera(c2w=c2w, img=imgs[idx])
        cameras.append(camera)

    return cameras
