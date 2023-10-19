import sys
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d

# loaders
from datasets.loaders.dtu import load_dtu
from datasets.loaders.pac_nerf import load_pac_nerf
from datasets.utils.geometry import rotation_matrix
from datasets.scenes.scene import Scene


def get_poses_all(cameras):
    poses = []
    for camera in cameras:
        poses.append(camera.get_pose())
    poses = torch.stack(poses, 0)
    return poses


def load_point_clouds(point_clouds_paths, max_nr_points=1000):
    """Loads point cloud from files.
    If the file is a mesh, points are its vertices.

    Args:
        point_clouds_paths: ordered list of point cloud file paths
        max_nr_points: maximum number of points to load

    Returns:
        point_clouds []: ordered list of (N, 3) numpy arrays
    """

    point_clouds = []
    for pc_path in point_clouds_paths:
        # if exists, load it
        if os.path.exists(pc_path):
            # if format is .ply or .obj
            if pc_path.endswith(".ply") or pc_path.endswith(".obj"):
                print("Loading point cloud from {}".format(pc_path))
                point_cloud = o3d.io.read_point_cloud(pc_path)
                points_3d = np.asarray(point_cloud.points)
                if points_3d.shape[0] > max_nr_points:
                    # downsample
                    random_idx = np.random.choice(
                        points_3d.shape[0], max_nr_points, replace=False
                    )
                    points_3d = points_3d[random_idx]
                    point_clouds.append(points_3d)
                print("Loaded {} points from mesh".format(points_3d.shape[0]))
            else:
                raise ValueError("Unsupported point cloud format")
        else:
            raise ValueError("Point cloud path {} does not exist".format(pc_path))

    return point_clouds


class MVDataset(Dataset):
    """Dataset class for all static multi-view datasets."""

    # self.cameras = []

    def __init__(
        self,
        dataset_name,
        scene_name,
        data_path,
        point_clouds_paths=[],
        split="train",  # "all", train", "test"
        use_every_for_test_split=8,
        train_test_no_overlap=True,
        load_with_mask=True,
        auto_orient_method="up",  # "up", "none"
        auto_center_method="none",  # "poses", "focus", "none"
        # auto_scale_poses=False,  # automatically scale the poses to fit in +/- 1 bounding box
        # downscale_factor=1
        device="cpu",
    ):
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        # self.auto_scale_poses = auto_scale_poses

        # check if path exists
        if not os.path.exists(data_path):
            print("ERROR: path does not exist")
            sys.exit()

        # load scene cameras
        if split not in ["all", "train", "test"]:
            raise ValueError("Unknown split value, use 'all', 'train' or 'test'")

        print(f"Loading {split} data")

        if self.dataset_name == "dtu":
            # load dtu
            cameras_all = load_dtu(
                data_path,
                load_with_mask=load_with_mask,
                # downscale_factor=downscale_factor
                device=device,
            )

        elif self.dataset_name == "nerf_synthetic":
            pass

        elif self.dataset_name == "pac_nerf":
            # load pac_nerf
            cameras_all = load_pac_nerf(
                data_path,
                n_cameras=11,
                load_with_mask=load_with_mask,
                device=device,
            )

        # elif self.dataset_name == "llff":
        #     pass
        # elif self.dataset_name == "tanks_and_temples":
        #     pass
        else:
            print("ERROR: dataset not supported")
            sys.exit()

        if len(point_clouds_paths) > 0:
            self.point_clouds = load_point_clouds(point_clouds_paths)
        else:
            self.point_clouds = []

        # # align and center poses
        # poses_all = get_poses_all(cameras_all)

        # transform = auto_orient_and_center_poses(
        #     poses_all,
        #     method=auto_orient_method,
        #     center_method=auto_center_method,
        # )

        # for camera in cameras_all:
        #     camera.concat_transform(transform)

        # # scale poses
        # if self.auto_scale_poses:
        #     scale_factor = float(
        #         torch.max(torch.abs(get_poses_all(cameras_all)[:, :3, 3]))
        #     )
        #     transform = torch.eye(4)
        #     transform[3, 3] = scale_factor
        #     for camera in cameras_all:
        #         camera.concat_transform(transform)

        if split == "all":
            self.cameras = cameras_all
        else:
            # split into train and test
            self.cameras = []
            if split == "train":
                if train_test_no_overlap:
                    for i, camera in enumerate(cameras_all):
                        if i % use_every_for_test_split != 0:
                            self.cameras.append(camera)
                else:
                    self.cameras = cameras_all
            if split == "test":
                for i, camera in enumerate(cameras_all):
                    if i % use_every_for_test_split == 0:
                        self.cameras.append(camera)

        print(f"Loaded {len(self.cameras)} cameras")

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        camera = self.cameras[idx]

        # TODO: rays_o, rays_d, gt_rgb, gt_mask
        # other possibile dataset dependent outputs: gt_depth, gt_normal, gt_albedo, gt_brdf, gt_visibility

        intrinsics = camera.intrinsics.to("cuda")
        pose = camera.get_pose().to("cuda")
        img = camera.img.to("cuda")
        mask = camera.mask.to("cuda")

        return intrinsics, pose, img, mask


# from nerfstudio


# def focus_of_attention(poses, initial_focus):
#     """Compute the focus of attention of a set of cameras. Only cameras
#     that have the focus of attention in front of them are considered.

#     Args:
#         poses: The poses to orient.
#         initial_focus: The 3D point views to decide which cameras are initially activated.

#     Returns:
#         The 3D position of the focus of attention.
#     """
#     # References to the same method in third-party code:
#     # https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L145
#     # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L197
#     active_directions = -poses[:, :3, 2:3]
#     active_origins = poses[:, :3, 3:4]
#     # initial value for testing if the focus_pt is in front or behind
#     focus_pt = initial_focus
#     # Prune cameras which have the current have the focus_pt behind them.
#     active = (
#         torch.sum(
#             active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)),
#             dim=-1,
#         )
#         > 0
#     )
#     done = False
#     # We need at least two active cameras, else fallback on the previous solution.
#     # This may be the "poses" solution if no cameras are active on first iteration, e.g.
#     # they are in an outward-looking configuration.
#     while torch.sum(active.int()) > 1 and not done:
#         active_directions = active_directions[active]
#         active_origins = active_origins[active]
#         # https://en.wikipedia.org/wiki/Line–line_intersection#In_more_than_two_dimensions
#         m = torch.eye(3) - active_directions * torch.transpose(active_directions, -2, -1)
#         mt_m = torch.transpose(m, -2, -1) @ m
#         focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
#         active = (
#             torch.sum(
#                 active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)),
#                 dim=-1,
#             )
#             > 0
#         )
#         if active.all():
#             # the set of active cameras did not change, so we're done.
#             done = True
#     return focus_pt


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
    # elif center_method == "focus":
    #     translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "up":
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        rotation = rotation_matrix(up.cpu().numpy(), np.array([0, 0, 1.0]))
        rotation = torch.from_numpy(rotation)
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
