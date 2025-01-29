from rich import print
import numpy as np
from pathlib import Path
import os.path as osp
import os
import json
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
from pycolmap import SceneManager
from mvdatasets.utils.images import image_to_numpy
from mvdatasets import Camera
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.utils.printing import print_error, print_warning, print_success
from mvdatasets.utils.loader_utils import rescale
from mvdatasets.geometry.common import rot_euler_3d_deg
from dataclasses import dataclass, asdict


def load(
    dataset_path: Path,
    scene_name: str,
    config: dict,
    verbose: bool = False,
):
    """flow3d data format loader.

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
        int: Number of frames per camera
        int: Number of sequence frames
        float: Frames per second
    """
    scene_path = dataset_path / scene_name
    splits = config["splits"]

    # Valid values for specific keys
    valid_values = {
        "subsample_factor": [1],
    }

    # Validate specific keys
    for key, valid in valid_values.items():
        if key in config and config[key] not in valid:
            raise ValueError(f"{key} {config[key]} must be a value in {valid}")

    # Debugging output
    if verbose:
        print("config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    #
    if config["subsample_factor"] > 1:
        subsample_factor = int(config["subsample_factor"])
    else:
        subsample_factor = 1

    # foad data from flow3d_preprocessed
    
    flow3d_dir = scene_path / "flow3d_preprocessed"
    
    if not os.path.exists(flow3d_dir):
        raise ValueError(f"flow3d_preprocessed directory {flow3d_dir} does not exist.")
    
    # rgb images
    images_path = scene_path / "rgb" / f"{subsample_factor}x"
    if not os.path.exists(images_path):
        raise ValueError(f"Images directory {images_path} does not exist.")
    
    # depth images
    depths_dir = flow3d_dir / "aligned_depth_anything_colmap" / f"{subsample_factor}x"
    if not os.path.exists(depths_dir):
        raise ValueError(f"Depth directory {depths_dir} does not exist.")
    
    # mask images
    masks_dir = flow3d_dir / "track_anything" / f"{subsample_factor}x"
    if not os.path.exists(masks_dir):
        raise ValueError(f"Mask directory {masks_dir} does not exist.")
    
    # covisible images
    covisible_dir = flow3d_dir / "covisible" / f"{subsample_factor}x" / "val"
    if not os.path.exists(covisible_dir):
        raise ValueError(f"Covisible directory {covisible_dir} does not exist.")
    
    # read colmap data

    colmap_dir = flow3d_dir / "colmap" / "sparse"
    if not os.path.exists(colmap_dir):
        colmap_dir = os.path.join(scene_path, "sparse")

    if not os.path.exists(colmap_dir):
        raise ValueError(f"COLMAP directory {colmap_dir} does not exist.")

    manager = SceneManager(str(colmap_dir), image_path=str(images_path))
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
    w2c_dict = {}
    camera_ids = []
    timestamps_dict = {}
    ids_dict = {}
    Ks_dict = {}
    imsize_dict = dict()  # width, height
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    
    pbar = tqdm(imdata, desc="metadata", ncols=100)
    for i, k in enumerate(pbar):
        #
        im = imdata[k]
        im_name = im.name
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate(
            [np.concatenate([rot, trans], 1), bottom], axis=0, dtype=np.float32
        )
        w2c_dict[im_name] = w2c
        
        # extract timestamp from image name
        timestamp = int(im_name.split("_")[1].split(".")[0])
        timestamps_dict[im_name] = timestamp

        # support different camera intrinsics
        camera_id = im.camera_id
        ids_dict[im_name] = camera_id
        camera_ids.append(camera_id)

        # camera intrinsics
        cam = manager.cameras[camera_id]
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        Ks_dict[im_name] = K
        imsize_dict[im_name] = (
            cam.width,  # subsample_factor,
            cam.height,  # subsample_factor
        )

    print(f"[COLMAP] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")

    if len(imdata) == 0:
        raise ValueError("No images found in COLMAP.")

    # Convert extrinsics to camera-to-world.
    c2w_dict = {}
    for k in w2c_dict:
        c2w_dict[k] = np.linalg.inv(w2c_dict[k])
        
    # dict to list of poses
    poses_all = [c2w_dict[k] for k in c2w_dict]
    
    # rescale (optional)
    scene_radius_mult, min_camera_distance, max_camera_distance = rescale(
        poses_all, to_distance=config["max_cameras_distance"]
    )
    scene_radius = max_camera_distance
    
    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rot = rot_euler_3d_deg(
        config["rotate_deg"][0], config["rotate_deg"][1], config["rotate_deg"][2]
    )
    global_transform[:3, :3] = scene_radius_mult * rot

    # local transform
    local_transform = np.eye(4)
    
    # Image names from COLMAP
    imgs_names = [imdata[k].name for k in imdata]
    # sort by name
    imgs_names = sorted(imgs_names)
    
    # need to load 1 image to get the size
    img_path = os.path.join(images_path, imgs_names[0])
    img_pil = Image.open(img_path)
    actual_width, actual_height = img_pil.size
    print("actual_width", actual_width, "actual_height", actual_height)

    # load extra.json
    with open(os.path.join(scene_path, "extra.json"), "r") as fp:
        extra_meta = json.load(fp)
        fps = extra_meta["fps"]
    print("fps", fps)

    # load images if needed
    rgbs_dict = {}
    masks_dict = {}
    depths_dict = {}
    covisible_dict = {}
    if not config["pose_only"]:

        pbar = tqdm(imgs_names, desc="loading images", ncols=100)
        for img_name in pbar:
                
            # rgb
            img_path = os.path.join(images_path, img_name)
            # load PIL image
            img_pil = Image.open(img_path)
            img_np = image_to_numpy(img_pil, use_uint8=True)
            rgb_np = img_np[..., :3]
            # print("rgb", rgb_np.shape)
            rgbs_dict[img_name] = rgb_np
            
            # mask
            if config["load_masks"]:
                mask_path = os.path.join(masks_dir, img_name)
                # load PIL image
                img_pil = Image.open(mask_path)
                mask_np = image_to_numpy(img_pil, use_uint8=True)  # (H, W)
                mask_np = mask_np[..., None]  # (H, W, 1)
                # print("mask", mask_np.shape)
                masks_dict[img_name] = mask_np
                
            # depth
            if config["load_depths"]:
                # change extension to .npy
                depth_name = img_name.replace(".png", ".npy")
                depth_path = os.path.join(depths_dir, depth_name)
                # check if file exists
                if not os.path.exists(depth_path):
                    # it is a test camera, no depth is given, just skip
                    pass
                else:
                    # load npy
                    depth_np = np.load(depth_path)  # (H, W)
                    depth_np = depth_np.astype(np.float32)
                    depth_np = depth_np[..., None]  # (H, W, 1)
                    # print("depth", depth_np.shape)
                    depths_dict[img_name] = depth_np

            # covisible
            if config["load_covisible"]:
                covisible_path = os.path.join(covisible_dir, img_name)
                # check if file exists
                if not os.path.exists(covisible_path):
                    # it is a train camera, no covisible is given, just skip
                    pass
                else:
                    # load PIL image
                    img_pil = Image.open(covisible_path)
                    covisible_np = image_to_numpy(img_pil, use_uint8=True)  # (H, W)
                    covisible_np = covisible_np[..., None]  # (H, W, 1)
                    # print("covisible", covisible_np.shape)
                    covisible_dict[img_name] = covisible_np

    # TODO: load 2D tracks
    if False:
    
        # Load 2D tracks
        frame_names = []

        # Load the query pixels from 2D tracks.
        query_tracks_2d = [
            np.load(
                osp.join(
                    flow3d_dir,
                    "/2d_tracks/",
                    f"{subsample_factor}x/{frame_name}_{frame_name}.npy",
                )
            ).astype(np.float32)
            for frame_name in frame_names
        ]

        num_samples = 1000
        num_frames = 10  # TODO: full sequence length
        step = 1

        raw_tracks_2d = []
        candidate_frames = list(range(0, num_frames, step))
        num_sampled_frames = len(candidate_frames)
        for i in tqdm(candidate_frames, desc="Loading 2D tracks", leave=False):
            curr_num_samples = query_tracks_2d[i].shape[0]
            num_samples_per_frame = (
                int(np.floor(num_samples / num_sampled_frames))
                if i != candidate_frames[-1]
                else num_samples
                - (num_sampled_frames - 1) * int(np.floor(num_samples / num_sampled_frames))
            )
            if num_samples_per_frame < curr_num_samples:
                track_sels = np.random.choice(
                    curr_num_samples, (num_samples_per_frame,), replace=False
                )
            else:
                track_sels = np.arange(0, curr_num_samples)

            curr_tracks_2d = []
            for j in range(0, num_frames, step):
                if i == j:
                    target_tracks_2d = query_tracks_2d[i]
                else:
                    target_tracks_2d = np.load(
                        osp.join(
                            scene_path,
                            "flow3d_preprocessed/2d_tracks/",
                            f"{subsample_factor}x/"
                            f"{frame_names[i]}_"
                            f"{frame_names[j]}.npy",
                        )
                    ).astype(np.float32)

                curr_tracks_2d.append(target_tracks_2d[track_sels])
            # stack with numpy
            raw_tracks_2d.append(np.stack(curr_tracks_2d, axis=1))
            
            # TODO: convert to 3D tracks
            # ...
    
    # cameras objects
    cameras_splits = {}
    
    if "train" in splits:
        cameras_splits["train"] = []
    
    if "val" in splits:
        cameras_splits["val"] = []
    
    pbar = tqdm(imgs_names, desc="loading images", ncols=100)
    for img_name in pbar:
        
        # check if img_name starts with "0"
        if not img_name.startswith("0"):
            # it is not a train camera, just skip
            split = "val"
        else:
            split = "train"
        
        # collect data
        c2w = c2w_dict.get(img_name)
        intrinsics = deepcopy(Ks_dict.get(img_name))
        timestamp = timestamps_dict.get(img_name)
        idx = ids_dict.get(img_name)
        colmap_width, colmap_height = imsize_dict.get(img_name)
        
        # check image scaling
        s_height = actual_height / colmap_height
        s_width = actual_width / colmap_width
        # intrinsics
        intrinsics[0, :] *= s_width
        intrinsics[1, :] *= s_height
        
        rgb_np = rgbs_dict.get(img_name)
        if rgb_np is not None:
            rgb_np = rgb_np[None, ...]  # (1, H, W, 3)
        mask_np = masks_dict.get(img_name)
        if mask_np is not None:
            mask_np = mask_np[None, ...]  # (1, H, W, 1)
        depth_np = depths_dict.get(img_name)
        if depth_np is not None:
            depth_np = depth_np[None, ...]  # (1, H, W, 1)
        covisible_np = covisible_dict.get(img_name)
        if covisible_np is not None:
            covisible_np = covisible_np[None, ...]  # (1, H, W, 1)
            
        # create camera object
        camera = Camera(
            intrinsics=intrinsics,
            pose=c2w,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=rgb_np,
            masks=mask_np,
            depths=depth_np,
            timestamps=timestamp,
            camera_label=str(idx),
            width=actual_width,
            height=actual_height,
            subsample_factor=1,  # int(config["subsample_factor"]),
            # verbose=verbose,
        )
        # add to list
        cameras_splits[split].append(camera)

    return {
        "scene_type": config["scene_type"],
        # "init_sphere_radius_mult": config["init_sphere_radius_mult"],
        # "foreground_scale_mult": config["foreground_scale_mult"],
        "point_clouds": [point_cloud],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_radius": scene_radius,
        "fps": fps,
        "nr_per_camera_frames": 1,
        "nr_sequence_frames": len(cameras_splits["train"]),
    }
