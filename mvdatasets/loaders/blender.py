from rich import print
from pathlib import Path
import os
import json
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from mvdatasets import Camera
from mvdatasets.utils.images import image_to_numpy
from mvdatasets.utils.geometry import (
    deg2rad,
    scale_3d,
    rot_x_3d,
    rot_y_3d,
    pose_local_rotation,
    pose_global_rotation,
    get_min_max_cameras_distances,
)
from mvdatasets.utils.images import image_uint8_to_float32, image_float32_to_uint8
from mvdatasets.utils.printing import print_error, print_warning


def load_blender(
    scene_path: Path, splits: list, config: dict = {}, verbose: bool = False
):
    """Blender data format loader.

    Args:
        scene_path (Path): Path to the dataset scene folder.
        splits (list): Splits to load (e.g., ["train", "test"]).
        config (dict): Dictionary of configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        cameras_splits (dict): Dictionary of splits with lists of Camera objects.
        global_transform (np.ndarray): (4, 4)
    """
    # Default configuration
    defaults = {
        "scene_type": "bounded",
        "load_mask": True,
        "use_binary_mask": True,
        "rotate_scene_x_axis_deg": 0.0,
        "subsample_factor": 1,
        "white_bg": True,
        "test_skip": 20,
        "scene_radius_mult": 0.5,
        "target_cameras_max_distance": 1.0,
        "init_sphere_scale": 0.3,
        "pose_only": False,
    }

    # Update config with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            if verbose:
                print(f"Setting '{key}' to default value: {default_value}")

    # Debugging output
    if verbose:
        print("load_blender config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    # read all poses
    poses_all = []
    for split in splits:
        # load current split transforms
        with open(os.path.join(scene_path, f"transforms_{split}.json"), "r") as fp:
            metas = json.load(fp)

        for frame in metas["frames"]:
            camera_pose = frame["transform_matrix"]
            poses_all.append(camera_pose)

    # find scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(poses_all)

    # define scene scale
    scene_scale = max_camera_distance * config["scene_radius_mult"]
    # round to 2 decimals
    scene_scale = round(scene_scale, 2)

    # scene scale such that furthest away camera is at target distance
    scene_scale_mult = config["target_cameras_max_distance"] / (
        max_camera_distance + 1e-2
    )

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    global_transform[:3, :3] = scene_scale_mult * rot_x_3d(
        deg2rad(rotate_scene_x_axis_deg)
    )

    # local transform
    local_transform = np.eye(4)
    local_transform[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # cameras objects
    height, width = None, None
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []

        # load current split transforms
        with open(os.path.join(scene_path, f"transforms_{split}.json"), "r") as fp:
            metas = json.load(fp)

        camera_angle_x = metas["camera_angle_x"]

        # load images to cpu as numpy arrays
        # (optional) load mask images to cpu as numpy arrays
        frames_list = []

        for frame in metas["frames"]:
            img_path = frame["file_path"].split("/")[-1]
            # check if file format is in the path
            if not img_path.endswith(".png"):
                img_path += ".png"
            camera_pose = frame["transform_matrix"]
            frames_list.append((img_path, camera_pose))
        frames_list.sort(key=lambda x: int(x[0].split(".")[0].split("_")[-1]))

        if split == "test":
            # skip every test_skip images
            test_skip = config["test_skip"]
            frames_list = frames_list[::test_skip]

        # iterate over images and load them
        pbar = tqdm(frames_list, desc=split, ncols=100)
        for frame in pbar:

            # get image name
            im_name = frame[0]
            # camera_pose = frame[1]

            # get frame idx and pose
            idx = int(frame[0].split(".")[0].split("_")[-1])

            if config["pose_only"]:

                # do not load images
                cam_imgs = None
                cam_masks = None

                # only read first image to get image size
                if height is None or width is None:
                    img_pil = Image.open(os.path.join(scene_path, f"{split}", im_name))
                    width, height = img_pil.size

            else:

                # load PIL image
                img_pil = Image.open(os.path.join(scene_path, f"{split}", im_name))
                img_np = image_to_numpy(img_pil, use_uint8=True)

                # TODO: subsample image
                # if subsample_factor > 1:
                #   subsample image

                # override H, W
                if height is None or width is None:
                    height, width = img_np.shape[:2]

                if config["load_mask"]:
                    # use alpha channel as mask
                    # (nb: this is only resonable for synthetic data)
                    mask_np = img_np[..., -1, None]
                    if config["use_binary_mask"]:
                        mask_np = mask_np > 0
                        mask_np = mask_np.astype(np.uint8) * 255
                else:
                    mask_np = None

                # apply white background, else black
                if config["white_bg"]:
                    if img_np.dtype == np.uint8:
                        # values in [0, 255], cast to [0, 1], run operation, cast back
                        img_np = image_uint8_to_float32(img_np)
                        img_np = img_np[..., :3] * img_np[..., -1:] + (
                            1 - img_np[..., -1:]
                        )
                        img_np = image_float32_to_uint8(img_np)
                    else:
                        # values in [0, 1]
                        img_np = img_np[..., :3] * img_np[..., -1:] + (
                            1 - img_np[..., -1:]
                        )
                else:
                    img_np = img_np[..., :3]

                # get images
                cam_imgs = img_np[None, ...]
                # print(cam_imgs.shape)

                # get mask (optional)
                if config["load_mask"]:
                    cam_masks = mask_np[None, ...]
                    # print(cam_masks.shape)
                else:
                    cam_masks = None

            pose = np.array(frame[1], dtype=np.float32)
            intrinsics = np.eye(3, dtype=np.float32)
            focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)
            intrinsics[0, 0] = focal_length
            intrinsics[1, 1] = focal_length
            intrinsics[0, 2] = width / 2.0
            intrinsics[1, 2] = height / 2.0

            camera = Camera(
                intrinsics=intrinsics,
                pose=pose,
                global_transform=global_transform,
                local_transform=local_transform,
                rgbs=cam_imgs,
                masks=cam_masks,
                camera_idx=idx,
                width=width,
                height=height,
                subsample_factor=int(config["subsample_factor"]),
            )

            cameras_splits[split].append(camera)

    return {
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "config": config,
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_scale": scene_scale,
        "scene_scale_mult": scene_scale_mult,
    }
