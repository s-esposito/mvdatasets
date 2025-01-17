from rich import print
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from mvdatasets.utils.printing import print_warning, print_success
from mvdatasets.camera import Camera
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.utils.images import image_to_numpy
from mvdatasets.geometry.common import deg2rad, rot_x_3d, get_min_max_cameras_distances
from PIL import Image


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str] = ["train", "test"],
    config: dict = {},
    verbose: bool = False,
):
    scene_path = dataset_path / scene_name

    # Default configuration
    defaults = {
        "scene_type": "bounded",
        "load_masks": True,
        "rotate_scene_x_axis_deg": -90.0,
        "subsample_factor": 1,
        "target_max_camera_distance": 1.0,
        "foreground_scale_mult": 1.0,
        "frame_rate": 30.0,
        "pose_only": False,
    }

    # Update config with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            if verbose:
                print_warning(f"Setting '{key}' to default value: {default_value}")
        else:
            if verbose:
                print_success(f"Using '{key}': {config[key]}")

    # Debugging output
    if verbose:
        print("load_panoptic_sports config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    # read all poses
    height, width = None, None
    temporal_dim = None
    c2w_split = {}
    intrisics_split = {}
    cam_ids_split = {}
    for split in ["train", "test"]:

        # load current split transforms
        with open(os.path.join(scene_path, f"{split}_meta.json"), "r") as fp:
            metas = json.load(fp)

        w2c_all = np.array(metas["w2c"], dtype=np.float32)  # (T, N, 4, 4)
        temporal_dim = w2c_all.shape[0]
        # drop temporal dimension, as the pose is the same for all frames
        w2c_all = w2c_all[0]

        c2w_all = np.linalg.inv(w2c_all)
        c2w_split[split] = c2w_all

        width = metas["w"]
        height = metas["h"]

        intrinsics_all = np.array(metas["k"], dtype=np.float32)  # (T, N, 3, 3)
        # drop temporal dimension, as the intrinsics are the same for all frames
        intrinsics_all = intrinsics_all[0]
        intrisics_split[split] = intrinsics_all

        # print("intrinsics_all:", intrinsics_all.shape)
        cam_ids = np.array(metas["cam_id"], dtype=np.int32)  # (N,)
        # remove temporal dimension
        cam_ids = cam_ids[0]
        cam_ids_split[split] = cam_ids  # (N,)

    # aggregate all poses
    poses_all = []
    for split in splits:
        poses_all.append(c2w_split[split])
    # concatenate poses_all over first dimension
    poses_all = np.concatenate(poses_all, axis=0)

    # find scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(poses_all)

    # scene scale such that furthest away camera is at target distance
    scene_radius_mult = config["target_max_camera_distance"] / max_camera_distance

    # new scene scale
    new_min_camera_distance = min_camera_distance * scene_radius_mult
    new_max_camera_distance = max_camera_distance * scene_radius_mult

    # scene radius
    scene_radius = new_max_camera_distance

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    global_transform[:3, :3] = scene_radius_mult * rot_x_3d(
        deg2rad(rotate_scene_x_axis_deg)
    )

    # load init point cloud "init_pt_cld.npz"
    point_cloud_path = scene_path / "init_pt_cld.npz"
    data = np.array(np.load(point_cloud_path)["data"], dtype=np.float32)

    point_clouds = []
    points_3d = data[:, :3]
    points_rgb = (data[:, 3:6] * 255.0).astype(np.uint8)
    fg_mask = data[:, -1]
    # make it a boolean mask
    fg_mask = fg_mask > 0.5
    # print("nr foreground points:", np.sum(fg_mask))
    # print("nr background points:", np.sum(~fg_mask))
    point_clouds.append(
        PointCloud(
            points_3d[~fg_mask],
            points_rgb[~fg_mask],
            label="background",
            marker="o",
        )
    )
    point_clouds.append(
        PointCloud(
            points_3d[fg_mask],
            points_rgb[fg_mask],
            label="foreground",
            marker="x",
        )
    )

    cam_timestamps = np.arange(temporal_dim) / config["frame_rate"]

    cameras_splits = {}
    for split in splits:

        cameras_splits[split] = []
        pbar = tqdm(cam_ids_split[split], desc=f"{split} cameras", ncols=100)
        for i, camera_id in enumerate(pbar):

            if not config["pose_only"]:

                # get all images in camera_id folder
                cam_path = scene_path / "ims" / str(camera_id)

                # list all images in the folder
                cam_imgs = []
                imgs_files = sorted(list(cam_path.glob("*.jpg")))
                # print(imgs_files)
                for img_file in imgs_files:
                    # load image
                    img_pil = Image.open(img_file)
                    img_np = image_to_numpy(img_pil, use_uint8=True)
                    cam_imgs.append(img_np)
                cam_imgs = np.stack(cam_imgs, axis=0)

                if config["load_masks"]:

                    # get all masks in camera_id folder
                    cam_path = scene_path / "seg" / str(camera_id)

                    # list all masks in the folder
                    cam_masks = []
                    masks_files = sorted(list(cam_path.glob("*.png")))
                    for mask_file in masks_files:
                        # load image
                        img_pil = Image.open(mask_file)
                        img_np = image_to_numpy(img_pil, use_uint8=True)[
                            ..., np.newaxis
                        ]
                        cam_masks.append(img_np)
                    # test cameras might not have masks
                    if len(cam_masks) > 0:
                        cam_masks = np.stack(cam_masks, axis=0)
                    else:
                        cam_masks = None
                else:
                    cam_masks = None

            else:
                cam_imgs = None
                cam_masks = None

            intrinsics = intrisics_split[split][i]
            pose = c2w_split[split][i]

            camera = Camera(
                intrinsics=intrinsics,
                pose=pose,
                global_transform=global_transform,
                # local_transform=local_transform,
                rgbs=cam_imgs,
                masks=cam_masks,
                timestamps=cam_timestamps,
                camera_label=camera_id,
                width=width,
                height=height,
                subsample_factor=int(config["subsample_factor"]),
                # verbose=verbose,
            )

            cameras_splits[split].append(camera)

    return {
        "scene_type": config["scene_type"],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "point_clouds": point_clouds,
        "min_camera_distance": new_min_camera_distance,
        "max_camera_distance": new_max_camera_distance,
        "foreground_scale_mult": config["foreground_scale_mult"],
        "scene_radius": scene_radius,
        "nr_per_camera_frames": temporal_dim,
        "fps": config["frame_rate"],
        "nr_sequence_frames": temporal_dim,
    }
