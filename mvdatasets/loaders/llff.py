from rich import print
import os
import numpy as np
import sys
import re
import pycolmap
import open3d as o3d
from tqdm import tqdm

from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.geometry import rot_x_3d, deg2rad
from mvdatasets.utils.pycolmap import read_points3D, read_cameras


def load_llff(
    scene_path,
    splits,
    config,
    verbose=False
):
    """llff data format loader

    Args:
        scene_path (str): path to the dataset scene folder
        splits (list): splits to load (e.g. ["train", "test"])
        config (dict): dict of config parameters

    Returns:
        cameras_splits (dict): dict of splits with lists of Camera objects
        global_transform (np.ndarray): (4, 4)
    """

    # CONFIG -----------------------------------------------------------------
        
    if "scene_type" not in config:
        config["scene_type"] = "unbounded"  # "forward_facing"
        if verbose:
            print(f"WARNING: scene_type not in config, setting to {config['scene_type']}")
    else:
        valid_scene_types = ["unbounded", "forward_facing"]
        if config["scene_type"] not in valid_scene_types:
            raise ValueError(f"scene_type {config['scene_type']} must be a value in {valid_scene_types}")
    
    if "rotate_scene_x_axis_deg" not in config:
        config["rotate_scene_x_axis_deg"] = 0.0
        if verbose:
            print(f"WARNING: rotate_scene_x_axis_deg not in config, setting to {config['rotate_scene_x_axis_deg']}")
    
    if "test_camera_freq" not in config:
        config["test_camera_freq"] = 8
        if verbose:
            print(f"WARNING: test_camera_freq not in config, setting to {config['test_camera_freq']}")
    
    if "train_test_overlap" not in config:
        config["train_test_overlap"] = False
        if verbose:
            print(f"WARNING: train_test_overlap not in config, setting to {config['train_test_overlap']}")
    
    if "scene_scale_mult" not in config:
        config["scene_scale_mult"] = 0.1
        if verbose:
            print(f"WARNING: scene_scale_mult not in config, setting to {config['scene_scale_mult']}")

    if "subsample_factor" not in config:
        config["subsample_factor"] = 1
        if verbose:
            print(f"WARNING: subsample_factor not in config, setting to {config['subsample_factor']}")
    else:
        valid_subsample_factors = [1, 2, 4, 8]
        if config["subsample_factor"] not in valid_subsample_factors:
            raise ValueError(f"subsample_factor {config['subsample_factor']} must be a value in {valid_subsample_factors}")
            
    if "scene_radius" not in config:
        config["scene_radius"] = 5.0
        if verbose:
            print(f"WARNING: scene_radius not in config, setting to {config['scene_radius']}")
        
    if verbose:
        print("load_llff config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")
        
    # -------------------------------------------------------------------------
    
    # global transform
    global_transform = np.eye(4)
    # rotate
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    rotation = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))
    # scale
    scene_scale_mult = config["scene_scale_mult"]
    s_rotation = scene_scale_mult * rotation
    global_transform[:3, :3] = s_rotation
    # scene radius
    scene_radius = config["scene_radius"] * scene_scale_mult
    
    # local transform
    local_transform = np.eye(4)
    # local_transform[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    
    # read colmap data
    
    reconstruction_path = os.path.join(scene_path, "sparse/0")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    # print(reconstruction.summary())

    point_cloud = read_points3D(reconstruction)  
    # # save point cloud as ply with o3d
    # o3d_point_cloud = o3d.geometry.PointCloud()
    # o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    # o3d.io.write_point_cloud(os.path.join("debug/point_clouds/mipnerf360", "garden.ply"), o3d_point_cloud)
    # exit()
    
    images_path = os.path.join(scene_path, "images")
    
    if config["subsample_factor"] > 1:
        subsample_factor = int(config["subsample_factor"])
        images_path += f"_{subsample_factor}"
    else:
        subsample_factor = 1
    
    cameras_meta = read_cameras(reconstruction, images_path)
    
    # # open poses_bounds.npy
    # poses_bounds_path = os.path.join(scene_path, "poses_bounds.npy")
    # # check if file exists
    # if os.path.exists(poses_bounds_path):
    #     poses_arr = np.load(poses_bounds_path)
    #     bounds = poses_arr[:, -2:]
    # else:
    #     bounds = np.array([0.01, 1.])
    
    # poses = []
    # for camera in cameras_meta:
    #     poses.append(camera["pose"])
    # poses = np.array(poses)
    
    # # TODO: forward_facing specific
    # if config["scene_type"] == "forward_facing":
    #     pass
    # else:
    #     # unbouded
    #     poses = unpad_poses(poses)
    #     # Rotate/scale poses to align ground with xy plane and fit to unit cube.
    #     poses, transform = transform_poses_pca(poses)
    #     poses = pad_poses(poses)

    #     print("transform", transform)
    #     print("poses", poses.shape)
        
    #     global_transform = transform
        
    # read images
    cameras_all = []
    # images_list = sorted(os.listdir(images_path), key=lambda x: int(re.search(r'\d+', x).group()))
    pbar = tqdm(cameras_meta, desc="images", ncols=100)
    for i, camera in enumerate(pbar):
        
        w2c = np.eye(4)
        w2c[:3, :3] = camera["rotation"]
        w2c[:3, 3] = camera["translation"]
        c2w = np.linalg.inv(w2c)
        pose = c2w
        
        # params
        params = camera["params"]
        intrinsics = np.eye(3)
        intrinsics[0, 0] = params["fx"] / subsample_factor
        intrinsics[1, 1] = params["fy"] / subsample_factor
        intrinsics[0, 2] = params["cx"] / subsample_factor
        intrinsics[1, 2] = params["cy"] / subsample_factor
        # print("intrinsics", intrinsics)
        
        idx = camera["id"]
        cam_imgs = camera["img"][None, ...]
        # print("cam_imgs", cam_imgs.shape)
        
        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
            params=params,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=cam_imgs,
            camera_idx=idx,
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
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "scene_radius": scene_radius,
        "scene_type": config["scene_type"],
        "point_clouds": [point_cloud],
        "config": config,  # TODO: find better way to return config settings
    }