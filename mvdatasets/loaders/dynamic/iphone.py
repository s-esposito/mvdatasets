from rich import print
import numpy as np
from pathlib import Path
import os
import json
from PIL import Image
from tqdm import tqdm
from mvdatasets.utils.images import image_to_numpy
from mvdatasets import Camera
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.utils.printing import print_error, print_warning, print_success
from mvdatasets.geometry.common import (
    deg2rad,
    scale_3d,
    rot_x_3d,
    rot_y_3d,
    get_min_max_cameras_distances,
)


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str] = ["train", "val"],
    config: dict = {},
    verbose: bool = False,
):
    """iphone data format loader.

    Args:
        dataset_path (Path): Path to the dataset folder.
        scene_name (str): Name of the scene / sequence to load.
        splits (list): Splits to load (e.g., ["train", "val"]).
        config (dict): Dictionary of configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        cameras_splits (dict): Dictionary of splits with lists of Camera objects.
        global_transform (np.ndarray): (4, 4)
    """
    scene_path = dataset_path / scene_name

    # Default configuration
    defaults = {
        "scene_type": "unbounded",
        # "load_masks": True,
        # "use_binary_mask": True,
        # "white_bg": True,
        "rotate_scene_x_axis_deg": 90.0,
        "subsample_factor": 1,
        "target_max_camera_distance": 1.0,
        # "foreground_radius_mult": 0.5,
        # "init_sphere_radius_mult": 0.3,
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
        print("load_blender config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------
    
    # load points.npy
    points_3d = np.load(os.path.join(scene_path, "points.npy"))
    point_cloud = PointCloud(points_3d, points_rgb=None)
    
    # load scene.json
    with open(os.path.join(scene_path, "scene.json"), "r") as fp:
        scene_meta = json.load(fp)
        near = scene_meta["near"]
        far = scene_meta["far"]
        scale = scene_meta["scale"]
        xyz = scene_meta["center"]
        center = np.array(xyz)
    print("near", near)
    print("far", far)
    print("scale", scale)
    print("center", center)
    
    # load extra.json
    with open(os.path.join(scene_path, "extra.json"), "r") as fp:
        extra_meta = json.load(fp)
        # vmin = np.array(extra_meta["bbox"][0])
        # vmax = np.array(extra_meta["bbox"][1])
        # center = vmin + (vmax - vmin) / 2
        # pose = np.eye(4)
        # pose[:3, 3] = center
        # local_scale = (vmax - vmin) / 2
        # bbox = BoundingBox(pose=pose, local_scale=local_scale)
        factor = extra_meta["factor"]
        fps = extra_meta["fps"]
        # lookat
        # up
    # print("bbox", bbox)
    print("factor", factor)
    print("fps", fps)
    
    # read splits data
    data = {}
    for split in splits:
        
        # 
        data[split] = {}
        
        # if split file not present, warning and skip 
        if not os.path.exists(os.path.join(scene_path, "splits", f"{split}.json")):
            print_warning(f"Split file not found: {os.path.join(scene_path, 'splits', f'{split}.json')}")
            continue
        
        # load current split data
        with open(os.path.join(scene_path, "splits", f"{split}.json"), "r") as fp:
            metas = json.load(fp)
            data[split]["camera_ids"] = np.array(metas["camera_ids"])
            data[split]["frame_names"] = metas["frame_names"]
            data[split]["timestamps"] = np.array(metas["time_ids"], dtype=np.float32) / fps

    poses_dict = {}
    intrinsics_dict = {}
    
    for split_name, split_data in data.items():
        
        poses_dict[split_name] = []
        intrinsics_dict[split_name] = []
        
        for frame_name in split_data["frame_names"]:
        
            # load current split transforms
            with open(os.path.join(scene_path, "camera", f"{frame_name}.json"), "r") as fp:
                frame_meta = json.load(fp)
                focal_length = frame_meta["focal_length"]
                image_size = frame_meta["image_size"]  # (W, H)
                orientation = np.array(frame_meta["orientation"])
                pixel_aspect_ratio = frame_meta["pixel_aspect_ratio"]
                position = np.array(frame_meta["position"])
                ## shift to scene center
                #position -= center
                ## unscale
                #position *= 1 / scale
                principal_point = np.array(frame_meta["principal_point"])
                radial_distortion = np.array(frame_meta["radial_distortion"])
                skew = frame_meta["skew"]
                tangential_distortion = np.array(frame_meta["tangential_distortion"])
                
                # assert all radial distortion values are extremely close to 0
                assert np.allclose(radial_distortion, 0.0, atol=1e-6)
                # assert tangential distortion values are extremely close to 0
                assert np.allclose(tangential_distortion, 0.0, atol=1e-6)
                # assert skew is 0
                assert np.allclose(skew, 0.0, atol=1e-6)
                # assert pixel aspect ratio is 1
                assert np.allclose(pixel_aspect_ratio, 1.0, atol=1e-6)

                # camera pose
                pose = np.eye(4)
                pose[:3, :3] = orientation
                pose[:3, 3] = position
                poses_dict[split_name].append(pose)
                # camera intrinsic matrix
                K = np.array([
                    [focal_length, skew, principal_point[0]],
                    [0, focal_length / pixel_aspect_ratio, principal_point[1]],
                    [0, 0, 1],
                ])
                intrinsics_dict[split_name].append(K)

    width, height = image_size
    
    # 
    poses_all = []
    for split_name, split_poses in poses_dict.items():
        for pose in split_poses:
            poses_all.append(pose)
    
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

    # local transform
    local_transform = np.eye(4)
    
    # load images if needed
    rgbs_dict = None
    masks_dict = None
    if not config["pose_only"]:
        
        rgbs_dict = {}
        pbar = tqdm(data.items(), desc="loading images", ncols=100)
        for split_name, split_data in pbar:
            rgbs_dict[split_name] = []
            
            frames_pbar = tqdm(split_data["frame_names"], desc=split_name, ncols=100)
            for frame_name in frames_pbar:
                rgb_path = os.path.join(scene_path, "rgb", "1x", f"{frame_name}.png")
                # load PIL image
                img_pil = Image.open(rgb_path)
                img_np = image_to_numpy(img_pil, use_uint8=True)
                # mask = img_np[..., -1]
                # # negate mask
                # mask = 255 - mask
                # rgb_np = img_np[..., :-1]
                # # set masked pixels to black
                # rgb_np[mask == 0] = 0
                # print("img_np", img_np.shape, img_np.dtype)
                # exit(0)
                rgb_np = img_np[..., :3]
                rgbs_dict[split_name].append(rgb_np)
                
        # if config["load_masks"]:
        #     masks_dict = {}
        #     for split_name, split_data in data.items():
        #         masks_dict[split_name] = []
    
    # cameras objects
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []
        
        for i, frame_name in enumerate(data[split]["frame_names"]):
            
            # print(i, frame_name)
            
            if rgbs_dict is not None:
                cam_imgs = rgbs_dict[split][i][None, ...]  # (1, H, W, 3)
            else:
                cam_imgs = None
                
            if masks_dict is not None:
                cam_masks = masks_dict[split][i]
            else:
                cam_masks = None
            
            # get camera id (int)
            idx = data[split]["camera_ids"][i]
            # get frame id (int)
            timestamp = data[split]["timestamps"][i]
            # get camera pose
            pose = poses_dict[split][i]
            # get camera intrinsics
            intrinsics = intrinsics_dict[split][i]
                
            camera = Camera(
                intrinsics=intrinsics,
                pose=pose,
                global_transform=global_transform,
                local_transform=local_transform,
                rgbs=cam_imgs,
                masks=cam_masks,
                timestamps=timestamp,
                camera_label=str(idx),
                width=width,
                height=height,
                subsample_factor=int(config["subsample_factor"]),
                # verbose=verbose,
            )
            
            cameras_splits[split].append(camera)
    
    return {
        "scene_type": config["scene_type"],
        # "init_sphere_radius_mult": config["init_sphere_radius_mult"],
        # "foreground_radius_mult": config["foreground_radius_mult"],
        "point_clouds": [point_cloud],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "min_camera_distance": new_min_camera_distance,
        "max_camera_distance": new_max_camera_distance,
        "scene_radius": scene_radius,
        "fps": fps,
        "nr_per_camera_frames": 1,
        "nr_sequence_frames": len(cameras_splits["train"]),
    }