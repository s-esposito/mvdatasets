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


def load_blender(
    scene_path,
    splits,
    config
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
    
    if "load_mask" in config:
        load_mask = config["load_mask"]
    else:
        load_mask = True
        
    if "rotate_scene_x_axis_deg" in config:
        rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    else:
        rotate_scene_x_axis_deg = -90
        
    if "scene_scale_mult" in config:
        scene_scale_mult = config["scene_scale_mult"]
    else:
        scene_scale_mult = 0.25
    
    # TODO: implement subsample_factor
    # if "subsample_factor" in config:
    #     subsample_factor = config["subsample_factor"]
    # else:
    #     subsample_factor = 1
    
    if "white_bg" in config:
        white_bg = config["white_bg"]
    else:
        white_bg = True
        
    if "test_skip" in config:
        test_skip = config["test_skip"]
    else:
        test_skip = 20
        
    # -------------------------------------------------------------------------
    
    height, width = None, None
    
    # global transform
    global_transform = np.eye(4)
    # rotate
    rotation = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))
    # scale
    s_rotation = scene_scale_mult * rotation
    global_transform[:3, :3] = s_rotation
    
    # local transform
    local_transform = np.eye(4)
    rotation = rot_x_3d(deg2rad(180))
    local_transform[:3, :3] = rotation
    
    # cameras objects
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []

        # load current split transforms
        with open(os.path.join(scene_path, f"transforms_{split}.json"), "r") as fp:
            metas = json.load(fp)
        
        camera_angle_x = metas["camera_angle_x"]
        
        # load images to cpu as numpy arrays
        # (optional) load mask images to cpu as numpy arrays
        imgs = []
        masks = []
        frames_list = []
        
        for frame in metas["frames"]:
            img_path = frame["file_path"].split('/')[-1] + '.png'
            camera_pose = frame["transform_matrix"]
            frames_list.append((img_path, camera_pose))
        frames_list.sort(key=lambda x: int(x[0].split('.')[0].split('_')[-1]))

        if split == 'test':
            # skip every test_skip images
            frames_list = frames_list[::test_skip]
        
        # iterate over images and load them
        pbar = tqdm(frames_list, desc=split, ncols=100)
        for frame in pbar:
            # get image name
            im_name = frame[0]
            camera_pose = frame[1]
            # load PIL image
            img_pil = Image.open(os.path.join(scene_path, f"{split}", im_name))
            img_np = image2numpy(img_pil)
            # TODO: subsample image
            # if subsample_factor != 1:
            #   subsample image
            
            # override H, W
            if height is None or width is None:
                height, width = img_np.shape[:2]
            
            if load_mask:
                # use alpha channel as mask
                # (nb: this is only resonable for synthetic data)
                mask_np = img_np[..., -1, None]
                masks.append(mask_np)
            
            # apply white background, else black
            if white_bg:
                img_np = img_np[..., :3] * img_np[..., -1:] + (1.0 - img_np[..., -1:])
            else:
                img_np = img_np[..., :3]
            
            imgs.append(img_np)

        # iterate over frames and create cameras
        for i, frame in enumerate(frames_list):
            # get frame idx and pose
            idx = int(frame[0].split('.')[0].split('_')[-1])
            
            # get images
            cam_imgs = imgs[i][None, ...]
            # print(cam_imgs.shape)
            
            # get mask (optional)
            if load_mask and len(masks) > i:
                cam_masks = masks[i][None, ...]
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
            )

            cameras_splits[split].append(camera)
    
    return cameras_splits, global_transform
