from rich import print
from pathlib import Path
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import cv2
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.utils.loader_utils import rescale
from mvdatasets.geometry.common import rot_euler_3d_deg
from mvdatasets.utils.printing import (
    print_error,
    print_warning,
    print_log,
    print_success,
)
from mvdatasets.geometry.quaternions import quats_to_rots
from mvdatasets import Camera
from mvdatasets.configs.dataset_config import DatasetConfig


def _generate_mask_from_polygons(
    annotations: list[dict],
    width: int,
    height: int,
):
    img = np.zeros((height, width, 1), dtype=np.uint8)
    for annotation in annotations:
        segments = annotation["segments"]
        polygons = []
        polygons.append(segments)
        ps = []
        for polygon in polygons:
            for poly in polygon:
                if poly == []:
                    poly = [[0.0, 0.0]]
                ps.append(np.array(poly, dtype=np.int32))
        cv2.fillPoly(img, ps, 255)
    return img


def _generate_semantic_mask_from_polygons(
    annotations: list[dict],
    width: int,
    height: int,
):
    img = np.zeros((height, width, 1), dtype=np.uint8)
    for annotation in annotations:
        class_id = annotation["class_id"]
        segments = annotation["segments"]
        polygons = []
        polygons.append(segments)
        ps = []
        for polygon in polygons:
            for poly in polygon:
                if poly == []:
                    poly = [[0.0, 0.0]]
                ps.append(np.array(poly, dtype=np.int32))
        cv2.fillPoly(img, ps, class_id)
    return img


def _extract_data(
    scene_name,
    annotations_path,
    frame_mapping,
    posed_images,
    sparse_annotations_path,
    split,
):

    # split either "train" or "val"
    if split not in ["train", "val"]:
        raise ValueError(f"Invalid split: {split}")

    if split == "val":
        # TODO: implement loading validation split
        # not implemented yet
        raise ValueError("Split 'val' not implemented/tested yet.")

    # read json file
    annotation_json_path = os.path.join(annotations_path, split, f"{scene_name}.json")
    if not os.path.exists(annotation_json_path):
        print_warning(
            f"JSON file {annotation_json_path} for split {split} does not exist."
        )

    f = open(annotation_json_path)
    # returns JSON object as a dictionary
    data = json.load(f)
    # close the file
    f.close()
    print_log(f"loaded {annotation_json_path}")

    # order frames (first to last)
    video_data = sorted(
        data["video_annotations"], key=lambda k: k["image"]["image_path"]
    )

    # load VISOR data
    frames_idxs = []
    # images_names = []
    mapped_images_names = []
    images_paths = []
    w2c_mats = []
    images_segments_dict = {}  # indexed by mapped image names
    pbar = tqdm(video_data, desc="frames", ncols=100)
    for frame_data in pbar:

        video_name = frame_data["image"]["video"]  # e.g.: P01_01
        video_name_prefix = video_name.split("_")[0]  # e.g.: P01
        image_name = frame_data["image"]["name"]  # e.g.: P01_01_frame_0000000140.jpg
        mapped_image_name = frame_mapping[image_name]  # e.g.: frame_0000000145.jpg

        # check if mapped image name is in JSON_DATA
        if mapped_image_name not in posed_images:
            print_warning(f"{mapped_image_name} not found in JSON_DATA, skipped.")
            continue

        # read camera pose
        pose_flat = posed_images[mapped_image_name]

        # QW, QX, QY, QZ, TX, TY, TZ
        quat = np.array(pose_flat[:4])
        # print("quat", quat.shape)
        rot = quats_to_rots(quat)
        # print(rot.shape)
        # exit(0)
        trasl = np.array(pose_flat[4:])
        w2c = np.eye(4)
        w2c[:3, :3] = rot
        w2c[:3, 3] = trasl
        w2c_mats.append(w2c)

        # get frame index
        # remove extension and get str format frame index and convert to int
        frame_idx = int(mapped_image_name.split(".")[0].split("_")[-1])
        frames_idxs.append(frame_idx)

        # get rgb image
        image_path = os.path.join(
            sparse_annotations_path, "rgb_frames", split, video_name_prefix, image_name
        )
        images_paths.append(image_path)

        # append image names
        # images_names.append(image_name)
        mapped_images_names.append(mapped_image_name)

        # get annotations
        images_segments_dict[mapped_image_name] = frame_data["annotations"]

    # Concatenate poses
    w2c_mats = np.stack(w2c_mats, axis=0)  # (N, 4, 4)
    c2w_mats = np.linalg.inv(w2c_mats)  # (N, 4, 4)

    # sorting poses based on frames number (first to last)
    frames_idxs = np.array(frames_idxs)  # (N,)
    inds = np.argsort(frames_idxs)
    c2w_mats = c2w_mats[inds]
    frames_idxs = frames_idxs[inds]

    # reorder images names and paths
    new_images_paths = []
    # new_images_names = []
    new_mapped_images_names = []
    for idx in inds:
        new_images_paths.append(images_paths[idx])
        # new_images_names.append(images_names[idx])
        new_mapped_images_names.append(mapped_images_names[idx])
    images_paths = new_images_paths
    # images_names = new_images_names
    mapped_images_names = new_mapped_images_names

    return (
        c2w_mats,
        frames_idxs,
        images_paths,
        mapped_images_names,
        images_segments_dict,
    )


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str],
    config: DatasetConfig,
    verbose: bool = False,
):
    """VISOR data format loader.

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

    config = config.asdict()  # Convert Config to dictionary

    # Valid values for specific keys
    valid_values = {}

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

    # load frame mapping
    # mapping of VISOR sparse frames to the originally released rgb_frames in EPIC-KITCHENS
    frame_mapping_path = dataset_path / "frame_mapping.json"
    with open(frame_mapping_path, "r") as fp:
        frame_mapping = json.load(fp)[scene_name]
    print_log(f"loaded frame mapping {frame_mapping_path}")

    # read JSON_DATA cameras
    cameras_path = dataset_path / "JSON_DATA"
    assert cameras_path.exists(), f"cameras_path path does not exist: {cameras_path}"

    cameras_json_path = cameras_path / f"{scene_name}.json"
    assert (
        cameras_json_path.exists()
    ), f"cameras_json_path path does not exist: {cameras_json_path}"

    with open(cameras_json_path, "r") as fp:
        sequence_data = json.load(fp)

    # load epic-kitchen sequence images data
    posed_images = sequence_data["images"]

    # load intrinsics
    camera = sequence_data["camera"]
    # camera_id = camera["id"]  # 1
    # camera_model = camera["model"]  # OPENCV
    focal_x = camera["params"][0]
    focal_y = camera["params"][1]
    c_x = camera["params"][2]
    c_y = camera["params"][3]
    K = np.array([[focal_x, 0, c_x], [0, focal_y, c_y], [0, 0, 1]], dtype=np.float32)
    # distortion coefficients
    # k_1 = camera["params"][4]
    # k_2 = camera["params"][5]
    # p_1 = camera["params"][6]
    # p_2 = camera["params"][7]
    camera_width = camera["width"]
    camera_height = camera["height"]
    target_width = 1920
    target_height = 1080
    scale_x = target_width / camera_width
    scale_y = target_height / camera_height
    K[0, :] *= scale_x
    K[1, :] *= scale_y

    # load point cloud
    points = sequence_data["points"]
    points_3d = []
    points_rgb = []
    for point in points:
        point_3d = np.array(point[:3])
        point_rgb = np.array(point[3:])
        points_3d.append(point_3d)
        points_rgb.append(point_rgb)
    points_3d = np.stack(points_3d, axis=0)
    points_rgb = np.stack(points_rgb, axis=0)
    point_cloud = PointCloud(points_3d, points_rgb=points_rgb)

    # load VISOR annotations
    sparse_annotations_path = dataset_path / "GroundTruth-SparseAnnotations"
    assert (
        sparse_annotations_path.exists()
    ), f"sparse_annotations_path path does not exist: {sparse_annotations_path}"

    # annotations paths
    annotations_path = sparse_annotations_path / "annotations"
    assert (
        annotations_path.exists()
    ), f"annotations_path path does not exist: {annotations_path}"

    # get train split (will be then splitted in train / test)
    split = "train"

    # TODO: load validation split

    res = _extract_data(
        scene_name,
        annotations_path,
        frame_mapping,
        posed_images,
        sparse_annotations_path,
        split,
    )
    c2w_mats = res[0]
    frames_idxs = res[1]
    images_paths = res[2]
    mapped_images_names = res[3]
    images_segments_dict = res[4]

    # rescale (optional)
    scene_radius_mult, min_camera_distance, max_camera_distance = rescale(
        c2w_mats, to_distance=config["max_cameras_distance"]
    )

    scene_radius = max_camera_distance

    # scene_transform = np.eye(4)

    # # scene rotation
    # rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    # scene_transform[:3, :3] = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))

    # # translate
    # translation_matrix = np.eye(4)
    # # translation_matrix[:3, 3] = [
    # #     config["translate_scene_x"],
    # #     config["translate_scene_y"],
    # #     config["translate_scene_z"],
    # # ]

    # # Incorporate translation into scene_transform
    # scene_transform = translation_matrix @ scene_transform

    # # Create scaling matrix
    # scaling_matrix = np.diag(
    #     [scene_radius_mult, scene_radius_mult, scene_radius_mult, 1]
    # )

    # # Incorporate scaling into scene_transform
    # scene_transform = scaling_matrix @ scene_transform

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rot = rot_euler_3d_deg(
        config["rotate_deg"][0], config["rotate_deg"][1], config["rotate_deg"][2]
    )
    global_transform[:3, :3] = scene_radius_mult * rot

    # local transform
    local_transform = np.eye(4)

    # apply global transform
    # point_cloud *= scene_radius_mult
    # point_cloud.transform(scene_transform)

    # build cameras
    cameras_all = []
    pbar = tqdm(
        zip(c2w_mats, images_paths, frames_idxs, mapped_images_names),
        desc="images",
        ncols=100,
    )
    for idx, camera_meta in enumerate(pbar):

        # unpack
        # c2w
        c2w_mat = camera_meta[0]
        # c2w_mat[:3, 3] *= scene_radius_mult
        # c2w_mat = scene_transform @ c2w_mat
        # img path
        img_path = camera_meta[1]
        # frame_index / frame_rate = time
        frame_idx = camera_meta[2]
        time = frame_idx / config["frame_rate"]
        cam_timestamp = np.array([time])

        # load img
        if config["pose_only"]:
            cam_imgs = None
        else:
            img_pil = Image.open(img_path)
            img_np = np.array(img_pil)[..., :3]
            cam_imgs = img_np[None, ...]  # (1, H, W, 3)

        # get annotations
        mapped_image_name = camera_meta[3]
        annotations = images_segments_dict[mapped_image_name]

        # get mask
        if not config["pose_only"] and config["load_masks"]:
            mask_np = _generate_mask_from_polygons(
                annotations=annotations, width=target_width, height=target_height
            )  # (H, W, 1)
            cam_masks = mask_np[None, ...]  # (1, H, W, 1)
        else:
            cam_masks = None

        # get semantic mask
        if not config["pose_only"] and config["load_semantic_masks"]:
            semantic_mask_np = _generate_semantic_mask_from_polygons(
                annotations=annotations, width=target_width, height=target_height
            )  # (H, W, 1)
            cam_semantic_masks = semantic_mask_np[None, ...]  # (1, H, W, 1)
        else:
            cam_semantic_masks = None

        # create camera
        camera = Camera(
            intrinsics=K,
            pose=c2w_mat,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=cam_imgs,
            masks=cam_masks,
            semantic_masks=cam_semantic_masks,
            timestamps=cam_timestamp,
            camera_label=frame_idx,
            height=target_height,
            width=target_width,
            subsample_factor=int(config["subsample_factor"]),
            # verbose=verbose,
        )

        cameras_all.append(camera)

    # split cameras into train and test
    # train_test_overlap = config["train_test_overlap"]
    # test_camera_freq = config["test_camera_freq"]

    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []
        # if split == "train":
        #     if train_test_overlap:
        # if train_test_overlap, use all cameras for training
        cameras_splits[split] = cameras_all
        #     # else use only a subset of cameras
        #     else:
        #         for i, camera in enumerate(cameras_all):
        #             if i % test_camera_freq != 0:
        #                 cameras_splits[split].append(camera)
        # if split == "test":
        #     # select a test camera every test_camera_freq cameras
        #     for i, camera in enumerate(cameras_all):
        #         if i % test_camera_freq == 0:
        #             cameras_splits[split].append(camera)

    return {
        "scene_type": config["scene_type"],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "point_clouds": [point_cloud],
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_radius": scene_radius,
        "foreground_scale_mult": config["foreground_scale_mult"],
        "nr_per_camera_frames": 1,
        "fps": config["frame_rate"],
        "nr_sequence_frames": len(cameras_splits["train"]),
    }
