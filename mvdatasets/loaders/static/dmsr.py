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
from mvdatasets.utils.loader_utils import rescale
from mvdatasets.geometry.common import rot_euler_3d_deg
from mvdatasets.utils.printing import print_error, print_warning, print_success


def load(
    dataset_path: Path,
    scene_name: str,
    config: dict,
    verbose: bool = False,
):
    """DMSR data format loader.

    Args:
        dataset_path (Path): Path to the dataset folder.
        scene_name (str): Name of the scene / sequence to load.
        splits (list): Splits to load (e.g., ["train", "val"]).
        config (DatasetConfig): Dataset configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        dict: Dictionary of splits with lists of Camera objects.
        np.ndarray: Global transform (4, 4)
        str: Scene type
        List[PointCloud]: List of PointClouds
        float: Minimum camera distance
        float: Maximum camera distance
        float: Foreground scale multiplier
        float: Scene radius
    """

    scene_path = dataset_path / scene_name
    splits = config["splits"]

    # Valid values for specific keys
    valid_values = {}

    # Validate specific keys
    for key, valid in valid_values.items():
        if key in config and config[key] not in valid:
            raise ValueError(f"{key} {config[key]} must be a value in {valid}")

    # Debugging output
    if verbose:
        print("config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    # read all poses
    poses_all = []
    for split in ["train", "test"]:
        # load current split transforms
        with open(os.path.join(scene_path, split, f"transforms.json"), "r") as fp:
            metas = json.load(fp)

        for frame in metas["frames"]:
            camera_pose = frame["transform_matrix"]
            poses_all.append(camera_pose)

    # rescale (optional)
    scene_radius_mult, min_camera_distance, max_camera_distance = rescale(
        poses_all, to_distance=config["max_cameras_distance"]
    )

    scene_radius = max_camera_distance

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rot = rot_euler_3d_deg(
        config["rotate_deg"][0], config["rotate_deg"][1], config["rotate_deg"][2]
    )
    global_transform[:3, :3] = scene_radius_mult * rot

    # local transform
    local_transform = np.eye(4)
    local_transform[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # cameras objects
    height, width = None, None
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []

        # load current split transforms
        with open(os.path.join(scene_path, split, f"transforms.json"), "r") as fp:
            metas = json.load(fp)

        camera_angle_x = metas["camera_angle_x"]

        # load images to cpu as numpy arrays
        frames_list = []

        for frame in metas["frames"]:
            img_path = frame["file_path"].split("/")[-1] + ".png"
            camera_pose = frame["transform_matrix"]
            frames_list.append((img_path, camera_pose))
        frames_list.sort(key=lambda x: int(x[0].split(".")[0].split("_")[-1]))

        if split == "test":
            # skip every test_skip images
            test_skip = config["test_skip"]
            frames_list = frames_list[::test_skip]

        # read H, W
        if height is None or width is None:
            frame = frames_list[0]
            im_name = frame[0]
            # load PIL image
            img_pil = Image.open(os.path.join(scene_path, f"{split}", "rgbs", im_name))
            img_np = image_to_numpy(img_pil, use_uint8=True)
            height, width = img_np.shape[:2]

        # iterate over images and load them
        pbar = tqdm(frames_list, desc=split, ncols=100)
        for frame in pbar:
            # get image name
            im_name = frame[0]
            # camera_pose = frame[1]

            if config["pose_only"]:
                cam_imgs = None
            else:
                # load PIL image
                img_pil = Image.open(
                    os.path.join(scene_path, f"{split}", "rgbs", im_name)
                )
                img_np = image_to_numpy(img_pil, use_uint8=True)
                # remove alpha (it is always 1)
                img_np = img_np[:, :, :3]
                # get images
                cam_imgs = img_np[None, ...]
                # depth_imgs = depth_np[None, ...]

            # im_name = im_name.replace('r', 'd')
            # depth_pil = Image.open(os.path.join(scene_path, f"{split}", "depth", im_name))
            # depth_np = image_to_numpy(depth_pil)[..., None]

            # get frame idx and pose
            idx = int(frame[0].split(".")[0].split("_")[-1])

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
                # depths=depth_imgs,
                masks=None,  # dataset has no masks
                camera_label=str(idx),
                width=width,
                height=height,
                subsample_factor=int(config["subsample_factor"]),
                # verbose=verbose,
            )

            cameras_splits[split].append(camera)

    return {
        "scene_type": config["scene_type"],
        "init_sphere_radius_mult": config["init_sphere_radius_mult"],
        "foreground_scale_mult": config["foreground_scale_mult"],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_radius": scene_radius,
    }
