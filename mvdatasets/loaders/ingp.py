from rich import print
import os
import json
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.images import image2numpy
from mvdatasets.utils.geometry import (
    deg2rad,
    scale_3d,
    rot_x_3d,
    rot_y_3d,
    pose_local_rotation,
    pose_global_rotation,
)
from mvdatasets.utils.images import (
    image_uint8_to_float32,
    image_float32_to_uint8
)


def load_ingp(
    scene_path,
    splits,
    config,
    verbose=False,
):
    """blender data format loader

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
        
    if "scene_scale_mult" not in config:
        config["scene_scale_mult"] = 0.25
        if verbose:
            print(f"WARNING: scene_scale_mult not in config, setting to {config['scene_scale_mult']}")
    
    if "subsample_factor" not in config:
        config["subsample_factor"] = 1
        if verbose:
            print(f"WARNING: subsample_factor not in config, setting to {config['subsample_factor']}")
        
    if "test_camera_freq" not in config:
        config["test_camera_freq"] = 8
        if verbose:
            print(f"WARNING: test_camera_freq not in config, setting to {config['test_camera_freq']}")
    
    if "train_test_overlap" not in config:
        config["train_test_overlap"] = False
        if verbose:
            print(f"WARNING: train_test_overlap not in config, setting to {config['train_test_overlap']}")

    if verbose:
        print("load_ingp config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")
        
    # -------------------------------------------------------------------------
    
    # cameras objects
    cameras_all = []
    
    # global transform
    global_transform = np.eye(4)
    # rotate
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    rotation = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))
    # scale
    scene_scale_mult = config["scene_scale_mult"]
    s_rotation = scene_scale_mult * rotation
    global_transform[:3, :3] = s_rotation
    
    # local transform
    local_transform = np.eye(4)
    rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    local_transform[:3, :3] = rotation
    
    # load camera params
    with open(os.path.join(scene_path, "transforms.json"), "r") as fp:
        metas = json.load(fp)
    
    # height, width = metas["h"], metas["w"]
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = metas["fl_x"]
    intrinsics[1, 1] = metas["fl_y"]
    intrinsics[0, 2] = metas["cx"]
    intrinsics[1, 2] = metas["cy"]
    scene_radius = metas["aabb_scale"] * config["scene_scale_mult"]
    
    for frame in metas["frames"]:
        
        pose = np.array(frame["transform_matrix"], dtype=np.float32)
        img_path = os.path.join(scene_path, frame["file_path"])
        
        # check if file exists
        if not os.path.exists(img_path):
            print(f"WARNING: {img_path} does not exist")
            continue
        
        idx = int(img_path.split('/')[-1].split('.')[0])
        
        # load PIL image
        img_pil = Image.open(img_path)
        img_np = image2numpy(img_pil, use_uint8=True)[:, :, :3]
                
        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=img_np[None, ...],
            masks=None,
            camera_idx=idx,
            subsample_factor=int(config["subsample_factor"]),
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