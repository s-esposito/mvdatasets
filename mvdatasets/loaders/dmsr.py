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
    rot_z_3d,
    pose_local_rotation,
    pose_global_rotation,
)


def load_dmsr(
    scene_path,
    splits,
    config,
    verbose=False,
):
    """dmsr data format loader

    Args:
        scene_path (str): path to the dataset scene folder
        splits (list): splits to load (e.g. ["train", "test"])
        config (dict): dict of config parameters

    Returns:
        cameras_splits (dict): dict of splits with lists of Camera objects
        global_transform (np.ndarray): (4, 4)
    """
    
    # CONFIG -----------------------------------------------------------------
    
    if "load_depth" not in config:
        config["load_depth"] = False
        if verbose:
            print(f"WARNING: load_depth not in config, setting to {config['load_depth']}")
    else:
        if config["load_depth"]:
            raise NotImplementedError("load_depth is not implemented yet")
        
    if "load_semantics" not in config:
        config["load_semantics"] = False
        if verbose:
            print(f"WARNING: load_semantics not in config, setting to {config['load_semantics']}")
    else:
        if config["load_semantics"]:
            raise NotImplementedError("load_semantics is not implemented yet")
        
    if "load_semantic_instance" not in config:
        config["load_semantic_instance"] = False
        if verbose:
            print(f"WARNING: load_semantic_instance not in config, setting to {config['load_semantic_instance']}")
    else:
        if config["load_semantic_instance"]:
            raise NotImplementedError("load_semantic_instance is not implemented yet")
        
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
        
    if "test_skip" not in config:
        config["test_skip"] = 1
        if verbose:
            print(f"WARNING: test_skip not in config, setting to {config['test_skip']}")
        
    if "scene_radius" not in config:
        config["scene_radius"] = 2.0
        if verbose:
            print(f"WARNING: scene_radius not in config, setting to {config['scene_radius']}")
        
    if verbose:
        print("load_dmsr config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")
        
    # -------------------------------------------------------------------------
    
    height, width = None, None
    
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
    # scene radius
    scene_radius = config["scene_radius"] * scene_scale_mult
    
    # cameras objects
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
            img_path = frame["file_path"].split('/')[-1] + '.png'
            camera_pose = frame["transform_matrix"]
            frames_list.append((img_path, camera_pose))
        frames_list.sort(key=lambda x: int(x[0].split('.')[0].split('_')[-1]))
        
        if split == 'test':
            # skip every test_skip images
            test_skip = config["test_skip"]
            frames_list = frames_list[::test_skip]
        
        # iterate over images and load them
        pbar = tqdm(frames_list, desc=split, ncols=100)
        for frame in pbar:
            # get image name
            im_name = frame[0]
            # camera_pose = frame[1]
            # load PIL image
            img_pil = Image.open(os.path.join(scene_path, f"{split}", "rgbs", im_name))
            img_np = image2numpy(img_pil, use_uint8=True)
            
            # remove alpha (it is always 1)
            img_np = img_np[:, :, :3]
            
            # im_name = im_name.replace('r', 'd')
            # depth_pil = Image.open(os.path.join(scene_path, f"{split}", "depth", im_name))
            # depth_np = image2numpy(depth_pil)[..., None]
            
            # override H, W
            if height is None or width is None:
                height, width = img_np.shape[:2]
            
            # get frame idx and pose
            idx = int(frame[0].split('.')[0].split('_')[-1])
            
            # get images
            cam_imgs = img_np[None, ...]
            # depth_imgs = depth_np[None, ...]
        
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
                camera_idx=idx,
                subsample_factor=int(config["subsample_factor"]),
            )

            cameras_splits[split].append(camera)
    
    return {
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "scene_radius": scene_radius,
        "config": config,
    }
