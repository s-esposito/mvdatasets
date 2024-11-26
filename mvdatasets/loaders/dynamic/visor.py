from rich import print
from pathlib import Path
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import cv2
from mvdatasets.geometry.common import rot_x_3d, deg2rad, get_min_max_cameras_distances
from mvdatasets.geometry.common import apply_transformation_3d
from mvdatasets.utils.printing import print_error, print_warning, print_log
from mvdatasets.geometry.quaternions import quats_to_rots
from mvdatasets import Camera


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


def _extract_data(scene_name, annotations_path, frame_mapping, posed_images, sparse_annotations_path, split):
    
    # split either "train" or "val"
    if split not in ["train", "val"]:
        print_error(f"Invalid split: {split}")
        
    # read json file
    annotation_json_path = os.path.join(annotations_path, split, f"{scene_name}.json")
    if not os.path.exists(annotation_json_path):
        print_warning(f"JSON file {annotation_json_path} for split {split} does not exist.")
    
    f = open(annotation_json_path)
    # returns JSON object as a dictionary
    data = json.load(f)
    # close the file
    f.close()
    print_log(f"loaded {annotation_json_path}")
    
    # order frames (first to last)
    video_data = sorted(data["video_annotations"], key=lambda k: k["image"]["image_path"])
    
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
        image_path = os.path.join(sparse_annotations_path, "rgb_frames", split, video_name_prefix, image_name)
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
    
    return c2w_mats, frames_idxs, images_paths, mapped_images_names, images_segments_dict


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str] = ["train", "val"],
    config: dict = {},
    verbose: bool = False
):
    """VISOR data format loader.

    Args:
        dataset_path (Path): Path to the dataset folder.
        scene_name (str): Name of the scene / sequence to load.
        splits (list): Splits to load (e.g., ["train", "test", "val"]).
        config (dict): Dictionary of configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        cameras_splits (dict): Dictionary of splits with lists of Camera objects.
        global_transform (np.ndarray): (4, 4)
    """
    
    # Default configuration
    defaults = {
        "scene_type": "unbounded",
        "translate_scene_x": 0.0,
        "translate_scene_y": -2.0,
        "translate_scene_z": 2.0,
        "rotate_scene_x_axis_deg": 90.0,
        "test_camera_freq": 50,
        "train_test_overlap": False,
        "subsample_factor": 1,
        "frame_rate": 59.94,
        "pose_only": False,
    }
    
    # Update config with defaults
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
            if verbose:
                print_warning(f"Setting '{key}' to default value: {default_value}")
    
    # Check for unimplemented features
    if config.get("pose_only"):
        if verbose:
            print_warning("pose_only is True, but this is not implemented yet")
    
    # Debugging output
    if verbose:
        print("load_blender config:")
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
    assert cameras_json_path.exists(), f"cameras_json_path path does not exist: {cameras_json_path}"
    
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
    for point in points:
        point_3d = np.array(point[:3])
        points_3d.append(point_3d)
    point_cloud = np.stack(points_3d, axis=0)
    print(point_cloud.shape)
    
    # load VISOR annotations
    sparse_annotations_path = dataset_path / "GroundTruth-SparseAnnotations"
    assert sparse_annotations_path.exists(), f"sparse_annotations_path path does not exist: {sparse_annotations_path}"
    
    # annotations paths
    annotations_path = sparse_annotations_path / "annotations"
    assert annotations_path.exists(), f"annotations_path path does not exist: {annotations_path}"
    
    # get train split (will be then splitted in train / test)
    split = "train"
    
    res = _extract_data(scene_name, annotations_path, frame_mapping, posed_images, sparse_annotations_path, split)
    c2w_mats = res[0]
    frames_idxs = res[1]
    images_paths = res[2]
    mapped_images_names = res[3]
    images_segments_dict = res[4]
    
    # find scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(c2w_mats)

    # scene scale such that furthest away camera is at target distance
    scene_radius_mult = 1.0

    # new scene scale
    new_min_camera_distance = min_camera_distance * scene_radius_mult
    new_max_camera_distance = max_camera_distance * scene_radius_mult

    # scene radius
    scene_radius = new_max_camera_distance

    # scene transform
    scene_transform = np.eye(4)
    # rotate and scale
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    scene_transform[:3, :3] = rot_x_3d(deg2rad(rotate_scene_x_axis_deg))
    # translate
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = [
        config["translate_scene_x"],
        config["translate_scene_y"],
        config["translate_scene_z"],
    ]
    scene_transform = translation_matrix @ scene_transform
    
    # global transform
    global_transform = np.eye(4)

    # local transform
    local_transform = np.eye(4)

    # apply global transform
    point_cloud *= scene_radius_mult
    point_cloud = apply_transformation_3d(point_cloud, scene_transform)
    
    # build cameras
    cameras_all = []
    pbar = tqdm(zip(c2w_mats, images_paths, frames_idxs, mapped_images_names), desc="images", ncols=100)
    for idx, camera_meta in enumerate(pbar):

        # unpack
        # c2w
        c2w_mat = camera_meta[0]
        c2w_mat[:3, 3] *= scene_radius_mult
        c2w_mat = scene_transform @ c2w_mat
        # img path
        img_path = camera_meta[1]
        # frame_index / frame_rate = time
        frame_idx = camera_meta[2]
        time = frame_idx / config["frame_rate"]
        cam_timestamp = np.array([time])

        # load img
        img_pil = Image.open(img_path)
        img_np = np.array(img_pil)[..., :3]
        cam_imgs = img_np[None, ...]  # (1, H, W, 3)
        
        # get annotations
        mapped_image_name = camera_meta[3]
        annotations = images_segments_dict[mapped_image_name]
        
        # get mask
        mask_np = _generate_mask_from_polygons(
            annotations=annotations,
            width=target_width,
            height=target_height
        ) # (H, W, 1)
        cam_masks = mask_np[None, ...]  # (1, H, W, 1)
        
        semantic_mask_np = _generate_semantic_mask_from_polygons(
            annotations=annotations,
            width=target_width,
            height=target_height
        ) # (H, W, 1)
        cam_semantic_masks = semantic_mask_np[None, ...]  # (1, H, W, 1)

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
            camera_idx=frame_idx,
            subsample_factor=int(config["subsample_factor"]),
            # verbose=verbose,
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
        "scene_type": config["scene_type"],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "point_clouds": [point_cloud],
        "min_camera_distance": new_min_camera_distance,
        "max_camera_distance": new_max_camera_distance,
        "scene_radius": scene_radius,
        "nr_per_camera_frames": 1,
        "nr_sequence_frames": len(cameras_splits["train"]),
    }