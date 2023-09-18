import sys
import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


from datasets.loaders.dtu import load_dtu


def get_poses_all(cameras):
    poses = []
    for camera in cameras:
        poses.append(camera.pose)
    poses = torch.stack(poses, 0)

    return poses


class MVStaticDataset(Dataset):
    """Dataset class for all static multi-view datasets."""

    # self.cameras = []

    def __init__(
        self,
        dataset_name,
        scene_name,
        data_path,
        split="train",  # "train", "test"
        test_every=8,
        train_test_no_overlap=True,
        load_with_mask=True,
        downscale_factor=1,
        # auto_scale_poses=False,
    ):
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        self.path = data_path
        # self.auto_scale_poses = auto_scale_poses

        # check if path exists
        if not os.path.exists(data_path):
            print("ERROR: path does not exist")
            sys.exit()

        # load scene cameras
        print(f"Loading {split} data")

        if self.dataset_name == "dtu":
            # load dtu
            cameras_all = load_dtu(
                data_path,
                load_with_mask=load_with_mask,
                # downscale_factor=downscale_factor,
            )

        elif self.dataset_name == "blender":
            pass
        elif self.dataset_name == "llff":
            pass
        elif self.dataset_name == "tanks_and_temples":
            pass
        else:
            print("ERROR: dataset not supported")
            sys.exit()

        # TODO: scale poses
        # scale_factor = 1.0
        # if self.auto_scale_poses:
        #     scale_factor /= float(torch.max(torch.abs(self.c2ws[:, :3, 3])))
        # self.c2ws[:, :3, 3] *= scale_factor

        # align and center poses
        transform = auto_orient_and_center_poses(
            get_poses_all(cameras_all), method="up", center_method="focus"
        )
        for camera in cameras_all:
            camera.concat_transform(transform)

        # split into train and test
        self.cameras = []
        if split == "train":
            if train_test_no_overlap:
                for i, camera in enumerate(cameras_all):
                    if i % test_every != 0:
                        self.cameras.append(camera)
            else:
                self.cameras = cameras_all
        if split == "test":
            for i, camera in enumerate(cameras_all):
                if i % test_every == 0:
                    self.cameras.append(camera)

        print(f"Loaded {len(self.cameras)} cameras")

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        camera = self.cameras[idx]
        return camera.intrinsics, camera.pose, camera.img, camera.mask


# from nerfstudio


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.torch.tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return (
        torch.eye(3)
        + skew_sym_mat
        + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))
    )


def focus_of_attention(poses, initial_focus):
    """Compute the focus of attention of a set of cameras. Only cameras
    that have the focus of attention in front of them are considered.

     Args:
        poses: The poses to orient.
        initial_focus: The 3D point views to decide which cameras are initially activated.

    Returns:
        The 3D position of the focus of attention.
    """
    # References to the same method in third-party code:
    # https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L145
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L197
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    # initial value for testing if the focus_pt is in front or behind
    focus_pt = initial_focus
    # Prune cameras which have the current have the focus_pt behind them.
    active = (
        torch.sum(
            active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)),
            dim=-1,
        )
        > 0
    )
    done = False
    # We need at least two active cameras, else fallback on the previous solution.
    # This may be the "poses" solution if no cameras are active on first iteration, e.g.
    # they are in an outward-looking configuration.
    while torch.sum(active.int()) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]
        # https://en.wikipedia.org/wiki/Line–line_intersection#In_more_than_two_dimensions
        m = torch.eye(3) - active_directions * torch.transpose(
            active_directions, -2, -1
        )
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = (
            torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
        )
        active = (
            torch.sum(
                active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)),
                dim=-1,
            )
            > 0
        )
        if active.all():
            # the set of active cameras did not change, so we're done.
            done = True
    return focus_pt


def auto_orient_and_center_poses(
    poses,
    method="up",  # "up", "none"
    center_method="focus",  # "poses", "focus", "none"
):
    """Orients and centers the poses.

    We provide three methods for orientation:

    - up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.

    There are two centering methods:

    - poses: The poses are centered around the origin.
    - focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    # translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "up":
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        rotation = rotation_matrix(up, torch.tensor([0, 0, 1], dtype=torch.float32))
        # transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        transform = torch.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ -translation
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return transform
