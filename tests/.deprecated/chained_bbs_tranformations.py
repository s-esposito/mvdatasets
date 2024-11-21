import numpy as np
import PIL
import os
import sys
import time
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import imageio

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.utils.profiler import Profiler
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.common import deg2rad, rot_x_3d, rot_y_3d, rot_z_3d
from mvdatasets.utils.point_clouds import load_point_cloud
from mvdatasets.geometry.common import apply_transformation_3d


def test():

    # load data

    data_path = "tests/assets/assetsnerf/chair.npy"
    data = np.load(data_path, allow_pickle=True).item()
    print(data["chair"].keys())

    # create bounding boxes
    bounding_boxes = []
    point_clouds = []

    bb_data = data["chair"].pop("bb")
    bb_pose = np.eye(4)
    bb_pose[:3, 3] = bb_data["center"]
    sizes = bb_data["dimensions"]
    father_bb = BoundingBox(
        pose=bb_pose,
        sizes=sizes,
        label="father_bb",
        color="orange",
        line_width=5.0,
    )
    bounding_boxes.append(father_bb)

    # load point cloud from ply
    points_3d = load_point_cloud("tests/assets/assetsnerf/5.ply")
    point_clouds.append(points_3d)

    # test save
    father_bb.save_as_ply(".", father_bb.label)

    for bb_key, bb_data in data["chair"].items():
        print("instance:", bb_key)
        bb = BoundingBox(
            pose=bb_data["transformation"],  # father to child transform
            father_bb=father_bb,
            label=bb_key,
            color="blue",
            line_width=5.0,
        )
        bounding_boxes.append(bb)
        # load instance point cloud (in world space)
        points_3d = load_point_cloud(f"tests/assets/assetsnerf/{bb_key}.ply")
        # align to father pc (in world space)
        points_3d = apply_transformation_3d(points_3d, bb_data["transformation"])
        point_clouds.append(points_3d)

    # bb_pose = np.eye(4)
    # bb_pose[:3, 3] = np.array([1.0, 0.0, 0.0])
    # bb_scale = np.array([0.7, 0.8, 0.9])
    # bb_pose[:3, :3] = rot_y_3d(deg2rad(45)) @ rot_x_3d(deg2rad(45))
    # bb_pose[:3, :3] *= bb_scale
    # bb = BoundingBox(
    #     pose=bb_pose,
    #     father_bb=father_bb,
    # )
    # bounding_boxes.append(bb)

    # bb_pose = np.eye(4)
    # bb_pose[:3, 3] = np.array([-0.5, 0.5, 0.0])
    # bb_scale = np.array([0.4, 0.3, 0.2])
    # bb_pose[:3, :3] = rot_y_3d(deg2rad(45)) @ rot_x_3d(deg2rad(45))
    # bb_pose[:3, :3] *= bb_scale
    # bb = BoundingBox(
    #     pose=bb_pose,
    #     father_bb=father_bb,
    # )
    # bounding_boxes.append(bb)

    # visualize bounding boxes
    fig = plot_3d(
        bounding_boxes=bounding_boxes,
        points_3d=np.concatenate(point_clouds, axis=0),
        azimuth_deg=90,
        elevation_deg=60,
        scene_radius=1.0,
        up="z",
        figsize=(15, 15),
        draw_origin=True,
        draw_frame=True,
        title="",
    )
    # plt.show()

    plt.savefig(
        os.path.join("plots", f"bbs.png"),
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":

    # Set a random seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(SEED)  # Set a random seed for GPU
    else:
        device = "cpu"
    torch.set_default_device(DEVICE)

    # Set default tensor type
    torch.set_default_dtype(torch.float32)

    test()
