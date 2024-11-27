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
from mvdatasets.geometry.common import rot_x_3d, deg2rad, get_min_max_cameras_distances
from mvdatasets.geometry.common import apply_transformation_3d
from mvdatasets.utils.printing import print_error, print_warning


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str] = ["train", "test"],
    config: dict = {},
    verbose: bool = False
):
    """LLFF data format loader.

    Args:
        dataset_path (Path): Path to the dataset folder.
        scene_name (str): Name of the scene / sequence to load.
        splits (list): Splits to load (e.g., ["train", "test"]).
        config (dict): Dictionary of configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        cameras_splits (dict): Dictionary of splits with lists of Camera objects.
        global_transform (np.ndarray): (4, 4)
    """
        
    scene_path = dataset_path / scene_name
    
    # Default configuration
    defaults = {
        "scene_type": "bounded",
        "translate_scene_x": 0.0,
        "translate_scene_y": 0.0,
        "translate_scene_z": 0.0,
        "rotate_scene_x_axis_deg": 0.0,
        "test_camera_freq": 8,
        "train_test_overlap": False,
        "subsample_factor": 1,
        "init_sphere_radius_mult": 0.1,
        "foreground_radius_mult": 0.5,
        "pose_only": False,
    }

    # Update config with defaults and handle warnings
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            if verbose:
                print_warning(f"{key} not in config, setting to {default_value}")

    # Valid values for specific keys
    valid_values = {
        "scene_type": ["bounded", "unbounded", "forward-facing"],
        "subsample_factor": [1, 2, 4, 8],
    }

    # Validate specific keys
    for key, valid in valid_values.items():
        if key in config and config[key] not in valid:
            print_error(f"{key} {config[key]} must be a value in {valid}")

    # Set `target_max_camera_distance` based on `scene_type`
    if config["scene_type"] == "bounded":
        config["target_max_camera_distance"] = 1.0
    elif config["scene_type"] == "unbounded":
        config["target_max_camera_distance"] = 0.5
    elif config["scene_type"] == "forward-facing":
        print_error("forward-facing scene type not implemented yet")

    # Check for unimplemented features
    if config.get("pose_only"):
        if verbose:
            print_warning("pose_only is True, but this is not implemented yet")

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
        print_error(f"Images directory {images_path} does not exist.")

    # read colmap data

    colmap_dir = os.path.join(scene_path, "sparse/0")
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(scene_path, "sparse")

    if not os.path.exists(colmap_dir):
        print_error(f"COLMAP directory {colmap_dir} does not exist.")

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
        print_error("No images found in COLMAP.")
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

    # TODO: test this on Bilarf dataset
    # # Load extended metadata. Used by Bilarf dataset.
    # extconf = {
    #     "spiral_radius_scale": 1.0,
    #     "no_factor_suffix": False,
    # }
    # extconf_file = os.path.join(scene_path, "ext_metadata.json")
    # if os.path.exists(extconf_file):
    #     with open(extconf_file) as f:
    #         extconf.update(json.load(f))

    # # TODO: forward-facing specific
    # # Load bounds if possible (only used in forward facing scenes).
    # if config["scene_type"] == "forward-facing":
    #     bounds = np.array([0.01, 1.0])
    #     posefile = os.path.join(scene_path, "poses_bounds.npy")
    #     if not os.path.exists(posefile):
    #         print_error(f"Pose bounds file {posefile} does not exist.")
    #     bounds = np.load(posefile)[:, -2:]

    # if config["scene_type"] == "forward-facing":
    #     pass
    # else:
    #     # unbouded
    #     poses = unpad_poses(poses)
    #     # Rotate/scale poses to align ground with xy plane and fit to unit cube.
    #     poses, transform = transform_poses_pca(poses)
    #     poses = pad_poses(poses)

    # load images
    # for d in [image_dir, colmap_image_dir]:
    #     if not os.path.exists(d):
    #         raise ValueError(f"Image folder {d} does not exist.")

    # # size of the scene measured by cameras
    # camera_locations = c2w_mats[:, :3, 3]
    # # scene_center = np.mean(camera_locations, axis=0)
    # # dists = np.linalg.norm(camera_locations - scene_center, axis=1)
    # dists = np.linalg.norm(camera_locations, axis=1)
    # scene_radius = np.max(dists)
    # print(f"Scene scale: {scene_radius:.3f}")

    # find scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(c2w_mats)

    # scene scale such that furthest away camera is at target distance
    scene_radius_mult = config["target_max_camera_distance"] / max_camera_distance

    # new scene scale
    new_min_camera_distance = min_camera_distance * scene_radius_mult
    new_max_camera_distance = max_camera_distance * scene_radius_mult

    # scene radius
    if config["scene_type"] == "bounded":
        scene_radius = new_max_camera_distance
    elif config["scene_type"] == "unbounded":
        scene_radius = 1.0

    scene_transform = np.eye(4)
    
    # scene rotation
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    scene_transform[:3, :3] = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))
    
    # scene translation
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = [
        config["translate_scene_x"],
        config["translate_scene_y"],
        config["translate_scene_z"],
    ]
    
    # Incorporate translation into scene_transform
    scene_transform = translation_matrix @ scene_transform
    
    # Create scaling matrix
    scaling_matrix = np.diag([scene_radius_mult, scene_radius_mult, scene_radius_mult, 1])

    # Incorporate scaling into scene_transform
    scene_transform = scaling_matrix @ scene_transform
    
    # global transform
    global_transform = np.eye(4)

    # local transform
    local_transform = np.eye(4)

    # apply global transform
    # point_cloud *= scene_radius_mult
    point_cloud.transform(scene_transform)

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
        img_path = os.path.join(images_path, img_name)
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil)[..., :3]
        actual_height, actual_width = img_np.shape[:2]
        cam_imgs = img_np[None, ...]  # (1, H, W, 3)
        # check image scaling
        s_height = actual_height / colmap_height
        s_width = actual_width / colmap_width
        # intrinsics
        intrinsics = deepcopy(Ks_dict[camera_id])
        intrinsics[0, :] *= s_width
        intrinsics[1, :] *= s_height

        # undistort
        if len(params) > 0:
            print_error("undistortion not implemented yet")

        # extrainsics
        c2w_mat = camera_meta[0]
        # c2w_mat[:3, 3] *= scene_radius_mult
        c2w_mat = scene_transform @ c2w_mat
        # create camera
        camera = Camera(
            intrinsics=intrinsics,
            pose=c2w_mat,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=cam_imgs,
            camera_idx=idx,
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
        "foreground_radius_mult": config["foreground_radius_mult"],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "point_clouds": [point_cloud],
        "min_camera_distance": new_min_camera_distance,
        "max_camera_distance": new_max_camera_distance,
        "scene_radius": scene_radius,
    }


def _undistort(camtype, params_dict, Ks_dict, imsize_dict, mask_dict):
    # undistortion
    Ks_dict = copy.deepcopy(Ks_dict)
    imsize_dict = copy.deepcopy(imsize_dict)
    mask_dict = copy.deepcopy(mask_dict)
    mapx_dict = dict()
    mapy_dict = dict()
    roi_undist_dict = dict()
    for camera_id in params_dict.keys():
        params = params_dict[camera_id]
        if len(params) == 0:
            continue  # no distortion
        assert camera_id in Ks_dict, f"Missing K for camera {camera_id}"
        assert camera_id in params_dict, f"Missing params for camera {camera_id}"
        K = Ks_dict[camera_id]
        width, height = imsize_dict[camera_id]

        if camtype == "perspective":
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            mask = None
        elif camtype == "fisheye":
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32),
                np.arange(height, dtype=np.float32),
                indexing="xy",
            )
            x1 = (grid_x - cx) / fx
            y1 = (grid_y - cy) / fy
            theta = np.sqrt(x1**2 + y1**2)
            r = (
                1.0
                + params[0] * theta**2
                + params[1] * theta**4
                + params[2] * theta**6
                + params[3] * theta**8
            )
            mapx = fx * x1 * r + width // 2
            mapy = fy * y1 * r + height // 2

            # Use mask to define ROI
            mask = np.logical_and(
                np.logical_and(mapx > 0, mapy > 0),
                np.logical_and(mapx < width - 1, mapy < height - 1),
            )
            y_indices, x_indices = np.nonzero(mask)
            y_min, y_max = y_indices.min(), y_indices.max() + 1
            x_min, x_max = x_indices.min(), x_indices.max() + 1
            mask = mask[y_min:y_max, x_min:x_max]
            K_undist = K.copy()
            K_undist[0, 2] -= x_min
            K_undist[1, 2] -= y_min
            roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
        else:
            print_error(f"Unknown camera type {camtype}")

        mapx_dict[camera_id] = mapx
        mapy_dict[camera_id] = mapy
        Ks_dict[camera_id] = K_undist
        roi_undist_dict[camera_id] = roi_undist
        imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
        mask_dict[camera_id] = mask

        return mapx_dict, mapy_dict, Ks_dict, roi_undist_dict, imsize_dict, mask_dict
