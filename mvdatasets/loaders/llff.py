from rich import print
import os
import numpy as np
import sys
import re
import pycolmap
from PIL import Image
import open3d as o3d
from tqdm import tqdm

from mvdatasets.utils.images import image2numpy
from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.geometry import qvec2rotmat, rot_x_3d, deg2rad


def read_points3D(reconstruction):
    point_cloud = []
    for point3D_id, point3D in reconstruction.points3D.items():
        point_cloud.append(point3D.xyz)
    point_cloud = np.array(point_cloud)
    return point_cloud


def read_cameras(reconstruction):
    
    intrinsics_all = {}
    for camera_id, camera in reconstruction.cameras.items():
        intrinsics = np.eye(3)
        print(camera.model_id)
        print(camera.params)
        # PINHOLE
        if camera.model_id == 1:
            intrinsics[0, 0] = camera.params[0]  # fx
            intrinsics[1, 1] = camera.params[1]  # fy
            intrinsics[0, 2] = camera.params[2]  # cx
            intrinsics[1, 2] = camera.params[3]  # cy
        # SIMPLE_RADIAL
        elif camera.model_id == 2:
            intrinsics[0, 0] = camera.params[0]  # fx
            intrinsics[1, 1] = camera.params[0]  # fy = fx
            intrinsics[0, 2] = camera.params[1]  # cx
            intrinsics[1, 2] = camera.params[2]  # cy
            # camera.params[3]  # k1
        else:
            raise NotImplementedError(f"camera model {camera.model_id} not implemented")
        print(intrinsics)
        intrinsics_all[str(camera_id)] = intrinsics
    
    extrinsics_all = {}
    for image_id, image in reconstruction.images.items():
        pose = np.eye(4)
        pose[:3, :3] = qvec2rotmat(image.qvec)
        pose[:3, 3] = image.tvec
        extrinsics_all[image.name] = pose
        intrinsics_all[image.name] = intrinsics_all[str(image.camera_id)]
    return intrinsics_all, extrinsics_all


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
        config["scene_scale_mult"] = 0.25
        if verbose:
            print(f"WARNING: scene_scale_mult not in config, setting to {config['scene_scale_mult']}")

    if "subsample_factor" not in config:
        config["subsample_factor"] = 1
        if verbose:
            print(f"WARNING: subsample_factor not in config, setting to {config['subsample_factor']}")
        
    if "scene_radius" not in config:
        config["scene_radius"] = 1.0
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
    
    # read colmap data
    
    reconstruction_path = os.path.join(scene_path, "sparse/0")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)

    # point_cloud = read_points3D(reconstruction)    
    # # save point cloud as ply with o3d
    # o3d_point_cloud = o3d.geometry.PointCloud()
    # o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    # o3d.io.write_point_cloud(os.path.join(scene_path, "point_cloud.ply"), o3d_point_cloud)
    # exit()
    
    intrinsics_all, extrinsics_all = read_cameras(reconstruction)
    images_path = os.path.join(scene_path, "images")
    
    if config["subsample_factor"] > 1:
        subsample_factor = int(config["subsample_factor"])
        images_path += f"_{subsample_factor}"
    else:
        subsample_factor = 1
        
    # local transform
    local_transform = np.eye(4)
    # local_transform[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    
    # read images and construct cameras
    cameras_all = []
    images_list = sorted(os.listdir(images_path), key=lambda x: int(re.search(r'\d+', x).group()))
    pbar = tqdm(images_list, desc="images", ncols=100)
    for im_name in pbar:
        
        # load PIL image
        img_pil = Image.open(os.path.join(images_path, im_name))
        img_np = image2numpy(img_pil, use_uint8=True)
        
        # params
        intrinsics = intrinsics_all[im_name]
        # update intrinsics after rescaling
        intrinsics[0, 0] *= 1/subsample_factor
        intrinsics[1, 1] *= 1/subsample_factor
        intrinsics[0, 2] *= 1/subsample_factor
        intrinsics[1, 2] *= 1/subsample_factor
        
        pose = extrinsics_all[im_name]
        cam_imgs = img_np[None, ...]
        idx = int(re.search(r'\d+', im_name).group())
        
        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
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
        "scene_radius": scene_radius
    }