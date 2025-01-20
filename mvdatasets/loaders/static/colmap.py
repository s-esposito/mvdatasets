from rich import print
from pathlib import Path
import os
import copy
import cv2
import numpy as np
from pycolmap import SceneManager
from tqdm import tqdm
from PIL import Image
from copy import deepcopy

from mvdatasets import Camera
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.utils.loader_utils import rescale
from mvdatasets.geometry.common import rot_euler_3d_deg
from mvdatasets.utils.printing import print_error, print_warning, print_success
from mvdatasets.configs.dataset_config import DatasetConfig


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str],
    config: DatasetConfig,
    verbose: bool = False,
):
    """LLFF data format loader.

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
    
    config = config.asdict()  # Convert Config to dictionary

    # Valid values for specific keys
    valid_values = {
        "scene_type": ["bounded", "unbounded"],
        "subsample_factor": [1, 2, 4, 8],
    }

    # Validate specific keys
    for key, valid in valid_values.items():
        if key in config and config[key] not in valid:
            raise ValueError(f"{key} {config[key]} must be a value in {valid}")

    # Set `max_cameras_distance` based on `scene_type`
    if config["scene_type"] == "bounded":
        config["max_cameras_distance"] = 1.0
    elif config["scene_type"] == "unbounded":
        config["max_cameras_distance"] = 0.5
    else:
        raise ValueError(f"Unknown scene type {config['scene_type']}")

    # Debugging output
    if verbose:
        print("load_colmap config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    # Images paths

    images_path = os.path.join(scene_path, "images")
    if config["subsample_factor"] > 1:
        subsample_factor = int(config["subsample_factor"])
        images_path += f"_{subsample_factor}"
    else:
        subsample_factor = 1

    if not os.path.exists(images_path):
        raise ValueError(f"Images directory {images_path} does not exist.")

    # read colmap data

    colmap_dir = os.path.join(scene_path, "sparse/0")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(scene_path, "sparse")

    if not os.path.exists(colmap_dir):
        raise ValueError(f"COLMAP directory {colmap_dir} does not exist.")

    manager = SceneManager(colmap_dir, image_path=images_path)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    # get points
    points_3d = manager.points3D.astype(np.float32)
    points_rgb = manager.point3D_colors
    # get points colors
    if points_rgb is not None:
        points_rgb = points_rgb.astype(np.uint8)
    point_cloud = PointCloud(points_3d, points_rgb)

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    camera_ids = []
    Ks_dict = dict()
    params_dict = dict()
    imsize_dict = dict()  # width, height
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

    pbar = tqdm(imdata, desc="metadata", ncols=100)
    for i, k in enumerate(pbar):
        #
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate(
            [np.concatenate([rot, trans], 1), bottom], axis=0, dtype=np.float32
        )
        w2c_mats.append(w2c)

        # support different camera intrinsics
        camera_id = im.camera_id
        camera_ids.append(camera_id)

        # camera intrinsics
        cam = manager.cameras[camera_id]
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        Ks_dict[camera_id] = K

        # Get distortion parameters.
        type_ = cam.camera_type
        if type_ == 0 or type_ == "SIMPLE_PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif type_ == 1 or type_ == "PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        if type_ == 2 or type_ == "SIMPLE_RADIAL":
            params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 3 or type_ == "RADIAL":
            params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 4 or type_ == "OPENCV":
            params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 5 or type_ == "OPENCV_FISHEYE":
            params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
            camtype = "fisheye"
        assert (
            camtype == "perspective" or camtype == "fisheye"
        ), f"Only perspective and fisheye cameras are supported, got {type_}"

        params_dict[camera_id] = params
        imsize_dict[camera_id] = (
            cam.width,  # subsample_factor,
            cam.height,  # subsample_factor
        )

    print(f"[COLMAP] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")

    if len(imdata) == 0:
        raise ValueError("No images found in COLMAP.")
    if not (type_ == 0 or type_ == 1):
        print_warning("COLMAP Camera is not PINHOLE. Images have distortion.")

    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    c2w_mats = np.linalg.inv(w2c_mats)

    # Image names from COLMAP
    imgs_names = [imdata[k].name for k in imdata]

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(imgs_names)
    imgs_names = [imgs_names[i] for i in inds]
    c2w_mats = c2w_mats[inds]
    camera_ids = [camera_ids[i] for i in inds]

    # rescale (optional)
    scene_radius_mult, min_camera_distance, max_camera_distance = rescale(
        c2w_mats, to_distance=config["max_cameras_distance"]
    )

    # scene radius
    if config["scene_type"] == "bounded":
        scene_radius = max_camera_distance
    elif config["scene_type"] == "unbounded":
        scene_radius = 1.0

    # scene_transform = np.eye(4)

    # # scene rotation
    # rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    # scene_transform[:3, :3] = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))

    # # scene translation
    # translation_matrix = np.eye(4)
    # # translation_matrix[:3, 3] = [
    # #     config["translate_scene_x"],
    # #     config["translate_scene_y"],
    # #     config["translate_scene_z"],
    # # ]

    # # Incorporate translation into scene_transform
    # scene_transform = translation_matrix @ scene_transform

    # # Create scaling matrix
    # scaling_matrix = np.diag(
    #     [scene_radius_mult, scene_radius_mult, scene_radius_mult, 1]
    # )

    # # Incorporate scaling into scene_transform
    # scene_transform = scaling_matrix @ scene_transform

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rot = rot_euler_3d_deg(
        config["rotate_deg"][0], config["rotate_deg"][1], config["rotate_deg"][2]
    )
    global_transform[:3, :3] = scene_radius_mult * rot

    # local transform
    local_transform = np.eye(4)

    # apply global transform
    # point_cloud *= scene_radius_mult
    # point_cloud.transform(global_transform)

    # need to load 1 image to get the size
    img_path = os.path.join(images_path, imgs_names[0])
    img_pil = Image.open(img_path)
    actual_width, actual_height = img_pil.size

    # build cameras
    cameras_all = []
    pbar = tqdm(zip(c2w_mats, camera_ids, imgs_names), desc="images", ncols=100)
    for idx, camera_meta in enumerate(pbar):

        # unpack
        camera_id = camera_meta[1]
        img_name = camera_meta[2]
        # get camera metadata
        params = params_dict[camera_id]
        colmap_width, colmap_height = imsize_dict[camera_id]

        # load img
        if not config["pose_only"]:
            img_path = os.path.join(images_path, img_name)
            img_pil = Image.open(img_path)
            img_np = np.array(img_pil)[..., :3]
            # actual_height, actual_width = img_np.shape[:2]
            cam_imgs = img_np[None, ...]  # (1, H, W, 3)
        else:
            cam_imgs = None

        # check image scaling
        s_height = actual_height / colmap_height
        s_width = actual_width / colmap_width
        # intrinsics
        intrinsics = deepcopy(Ks_dict[camera_id])
        intrinsics[0, :] *= s_width
        intrinsics[1, :] *= s_height

        # undistort
        if len(params) > 0:
            raise ValueError("undistortion not implemented yet")

        # extrainsics
        c2w_mat = camera_meta[0]
        # c2w_mat[:3, 3] *= scene_radius_mult
        # c2w_mat = global_transform @ c2w_mat
        # create camera
        camera = Camera(
            intrinsics=intrinsics,
            pose=c2w_mat,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=cam_imgs,
            camera_label=str(idx),
            width=actual_width,
            height=actual_height,
            subsample_factor=1,  # int(config["subsample_factor"]),
            # verbose=verbose,
        )

        cameras_all.append(camera)

    # split cameras into train and test
    train_test_overlap = config["train_test_overlap"]
    test_camera_freq = config["test_camera_freq"]
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []
        if split == "train":
            if train_test_overlap:
                # if train_test_overlap, use all cameras for training
                cameras_splits[split] = cameras_all
            # else use only a subset of cameras
            else:
                for i, camera in enumerate(cameras_all):
                    if i % test_camera_freq != 0:
                        cameras_splits[split].append(camera)
        if split == "test":
            # select a test camera every test_camera_freq cameras
            for i, camera in enumerate(cameras_all):
                if i % test_camera_freq == 0:
                    cameras_splits[split].append(camera)

    return {
        "scene_type": config["scene_type"],
        "init_sphere_radius_mult": config["init_sphere_radius_mult"],
        "foreground_scale_mult": config["foreground_scale_mult"],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "point_clouds": [point_cloud],
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_radius": scene_radius,
    }
