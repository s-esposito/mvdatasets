from rich import print
import numpy as np
from pathlib import Path
import os.path as osp
import os
import json
from PIL import Image
from tqdm import tqdm
from mvdatasets.utils.images import image_to_numpy
from mvdatasets import Camera
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.utils.printing import print_error, print_warning, print_success
from mvdatasets.utils.loader_utils import rescale
from mvdatasets.geometry.common import rot_euler_3d_deg
from dataclasses import dataclass, asdict
from mvdatasets.configs.dataset_config import DatasetConfig


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str],
    config: DatasetConfig,
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

    config = asdict(config)  # Convert Config to dictionary

    # Valid values for specific keys
    valid_values = {
        "subsample_factor": [1, 2],
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

    # TODO: load data from flow3d_preprocessed

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
            print_warning(
                f"Split file not found: {os.path.join(scene_path, 'splits', f'{split}.json')}"
            )
            continue

        # load current split data
        with open(os.path.join(scene_path, "splits", f"{split}.json"), "r") as fp:
            metas = json.load(fp)
            data[split]["camera_ids"] = np.array(metas["camera_ids"])
            data[split]["frame_names"] = metas["frame_names"]
            data[split]["timestamps"] = (
                np.array(metas["time_ids"], dtype=np.float32) / fps
            )

    poses_dict = {}
    intrinsics_dict = {}
    for split_name, split_data in data.items():

        poses_dict[split_name] = []
        intrinsics_dict[split_name] = []

        for frame_name in split_data["frame_names"]:

            # load current split transforms
            with open(
                os.path.join(scene_path, "camera", f"{frame_name}.json"), "r"
            ) as fp:
                frame_meta = json.load(fp)
                focal_length = frame_meta["focal_length"]
                image_size = frame_meta["image_size"]  # (W, H)
                orientation = np.array(frame_meta["orientation"])
                pixel_aspect_ratio = frame_meta["pixel_aspect_ratio"]
                position = np.array(frame_meta["position"])
                ## shift to scene center
                # position -= center
                ## unscale
                # position *= 1 / scale
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
                K = np.array(
                    [
                        [focal_length, skew, principal_point[0]],
                        [0, focal_length / pixel_aspect_ratio, principal_point[1]],
                        [0, 0, 1],
                    ]
                )
                intrinsics_dict[split_name].append(K)

    #
    if config["subsample_factor"] > 1:
        subsample_factor = int(config["subsample_factor"])
    else:
        subsample_factor = 1

    width, height = image_size
    width, height = width // subsample_factor, height // subsample_factor

    #
    poses_all = []
    for split_name, split_poses in poses_dict.items():
        for pose in split_poses:
            poses_all.append(pose)

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

    # load images if needed
    rgbs_dict = {}
    masks_dict = {}
    depths_dict = {}
    if not config["pose_only"]:

        pbar = tqdm(data.items(), desc="loading images", ncols=100)
        for split_name, split_data in pbar:
            rgbs_dict[split_name] = []

            frames_pbar = tqdm(split_data["frame_names"], desc=split_name, ncols=100)
            for frame_name in frames_pbar:
                rgb_path = os.path.join(
                    scene_path, "rgb", f"{subsample_factor}x", f"{frame_name}.png"
                )
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
        #     for split_name, split_data in data.items():
        #         masks_dict[split_name] = []

        if config["load_depths"]:
            for split_name, split_data in data.items():
                depths_dict[split_name] = []

                frames_pbar = tqdm(
                    split_data["frame_names"], desc=split_name, ncols=100
                )
                for frame_name in frames_pbar:
                    depth_path = os.path.join(
                        scene_path, "depth", f"{subsample_factor}x", f"{frame_name}.npy"
                    )
                    # check if file exists
                    if not os.path.exists(depth_path):
                        print_warning(f"Depth file not found: {depth_path}")
                        continue
                    # load npy
                    depth_np = np.load(depth_path)
                    # multiply depth times scene scale mult
                    depth_np *= scene_radius_mult
                    #
                    depths_dict[split_name].append(depth_np)

        # TODO: load covisible for validation split

    # Load 2D tracks
    frame_names = []

    # Load the query pixels from 2D tracks.
    query_tracks_2d = [
        np.load(
            osp.join(
                scene_path,
                "flow3d_preprocessed/2d_tracks/",
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

    # cameras objects
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []

        for i, frame_name in enumerate(data[split]["frame_names"]):

            # print(i, frame_name)
            rgbs_split = rgbs_dict.get(split)
            if rgbs_split is not None and len(rgbs_split) > 0:
                cam_imgs = rgbs_split[i][None, ...]  # (1, H, W, 3)
            else:
                cam_imgs = None

            masks_split = masks_dict.get(split)
            if masks_split is not None and len(masks_split) > 0:
                cam_masks = masks_split[i][None, ...]  # (1, H, W, 1)
            else:
                cam_masks = None

            depths_split = depths_dict.get(split)
            if depths_split is not None and len(depths_split) > 0:
                cam_depths = depths_split[i][None, ...]  # (1, H, W, 1)
            else:
                cam_depths = None

            # get camera id (int)
            idx = data[split]["camera_ids"][i]
            # get frame id (int)
            timestamp = data[split]["timestamps"][i]
            # get camera pose
            pose = poses_dict[split][i]
            # get camera intrinsics
            intrinsics = intrinsics_dict[split][i]

            # update intrinsics based on subsample factor
            intrinsics[0, :] *= 1 / float(subsample_factor)
            intrinsics[1, :] *= 1 / float(subsample_factor)

            camera = Camera(
                intrinsics=intrinsics,
                pose=pose,
                global_transform=global_transform,
                local_transform=local_transform,
                rgbs=cam_imgs,
                masks=cam_masks,
                depths=cam_depths,
                timestamps=timestamp,
                camera_label=str(idx),
                width=width,
                height=height,
                subsample_factor=1,  # int(config["subsample_factor"]),
                # verbose=verbose,
            )

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
