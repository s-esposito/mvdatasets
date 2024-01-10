import os
import numpy as np
import sys
import pycolmap

from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.geometry import qvec2rotmat


def read_cameras(reconstruction):
    intrinsics_all = {}
    for camera_id, camera in reconstruction.cameras.items():
        intrinsics = np.eye(3)
        intrinsics[0, 0] = camera.params[0]
        intrinsics[1, 1] = camera.params[1]
        intrinsics[0, 2] = camera.params[2]
        intrinsics[1, 2] = camera.params[3]
        intrinsics_all[camera_id] = intrinsics
    return intrinsics_all


def read_points3D(reconstruction):
    point_cloud = []
    for point3D_id, point3D in reconstruction.points3D.items():
        point_cloud.append(point3D.xyz)
    point_cloud = np.array(point_cloud)
    return point_cloud


def read_images(reconstruction):
    extrinsics_all = {}
    for image_id, image in reconstruction.images.items():
        pose = np.eye(4)
        pose[:3, :3] = qvec2rotmat(image.qvec)
        pose[:3, 3] = image.tvec
        extrinsics_all[image_id] = pose
    return extrinsics_all


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
        
    if "scene_scale_mult" not in config:
        if verbose:
            print("WARNING: scene_scale_mult not in config, setting to 0.25")
        config["scene_scale_mult"] = 0.25

    if "subsample_factor" not in config:
        if verbose:
            print("WARNING: subsample_factor not in config, setting to 1.0")
        config["subsample_factor"] = 1.0
        
    if "test_skip" not in config:
        if verbose:
            print("WARNING: test_skip not in config, setting to 20")
        config["test_skip"] = 20
        
    if verbose:
        print("load_llff config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")
        
    # -------------------------------------------------------------------------
    
    height, width = None, None
    
    # global transform
    global_transform = np.eye(4)
    # rotate
    rotation = np.eye(3)
    # scale
    scene_scale_mult = config["scene_scale_mult"]
    s_rotation = scene_scale_mult * rotation
    global_transform[:3, :3] = s_rotation
    
    # read colmap data
    
    reconstruction_path = os.path.join(scene_path, "sparse/0")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    intrinsics_all = read_cameras(reconstruction)
    assert len(intrinsics_all) == 1, "Only one camera is supported"

    point_cloud = read_points3D(reconstruction)
    print("point_cloud")
    print(point_cloud.shape)

    extrinsics_all = read_images(reconstruction)
    print("extrinsics")
    for extrinsics in extrinsics_all.values():
        print(extrinsics)
    
    images_path = os.path.join(scene_path, "images")
    
    if config["subsample_factor"] > 1.0:
        subsample_factor = int(config["subsample_factor"])
        images_path += f"_{subsample_factor}"
    
    # read images
    
    images = sorted(os.listdir(images_path))
    
    
    # local transform
    local_transform = np.eye(4)
    rotation = np.eye(3)
    local_transform[:3, :3] = rotation
    
    # cameras objects
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []
    
        for image in images:
    
            # WIP
    
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