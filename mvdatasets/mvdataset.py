import os
import torch
from mvdatasets.loaders.dtu import load_dtu
from mvdatasets.loaders.blender import load_blender
from mvdatasets.loaders.pac_nerf import load_pac_nerf
from mvdatasets.utils.point_clouds import load_point_clouds
from mvdatasets.utils.common import is_dataset_supported
# from mvdatasets.utils.bounding_primitives import Sphere, AABB

# from mvdatasets.utils.raycasting import get_camera_frames_per_points_2d
# from torch.utils.data import Dataset
# import open3d as o3d
# import numpy as np
# from mvdatasets.utils.geometry import rotation_matrix
# from mvdatasets.scenes.scene import Scene


class MVDataset:
    """Dataset class for all static multi-view datasets.

    All data is stored in CPU memory.
    """

    # self.cameras = []

    def __init__(
        self,
        dataset_name,
        scene_name,
        datasets_path,  # dataset root path
        splits=["train", "test"],  # ["train", "test"] or None (all)
        test_camera_freq=8,
        train_test_overlap=False,
        load_mask=True,
        point_clouds_paths=[],
        meshes_paths=[],
        # auto_orient_method="none",  # "up", "none"
        # auto_center_method="none",  # "poses", "focus", "none"
        # auto_scale_poses=False,  # scale the poses to fit in +/- 1 bounding box
        subsample_factor=1.0
        # profiler=None,
    ):
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        # self.per_camera_rays_batch_size = 512
        # self.profiler = profiler
        # self.auto_scale_poses = auto_scale_poses

        data_path = os.path.join(datasets_path, dataset_name, scene_name)

        # check if path exists
        if not os.path.exists(data_path):
            raise ValueError(f"ERROR: data path {data_path} does not exist")

        # load scene cameras
        if splits is None:
            splits = ["all"]
        elif "train" not in splits and "test" not in splits:
            raise ValueError(
                "ERROR: splits must contain at least one of 'train' or 'test'"
            )

        # check if dataset is supported
        if not is_dataset_supported(dataset_name):
            raise ValueError(f"ERROR: dataset {dataset_name} is not supported")

        print(f"loading {splits} splits")

        # load dtu
        if self.dataset_name == "dtu":
            cameras_all = load_dtu(data_path, load_mask=load_mask)

        # load nerf_synthetic
        elif self.dataset_name == "nerf_synthetic":
            cameras_all = load_blender(
                data_path, load_mask=load_mask
            )

        # load pac_nerf
        elif self.dataset_name == "pac_nerf":
            # TODO: find n_cameras automatically
            cameras_all = load_pac_nerf(
                                        data_path,
                                        n_cameras=11,
                                        load_mask=load_mask
                                    )

        # elif self.dataset_name == "llff":
        #     pass
        # elif self.dataset_name == "tanks_and_temples":
        #     pass

        # (optional) load point clouds
        if len(point_clouds_paths) > 0:
            self.point_clouds = load_point_clouds(point_clouds_paths)
            print(f"loaded {len(self.point_clouds)} point clouds")
        else:
            self.point_clouds = []
        
        # (optional) load meshes
        if len(meshes_paths) > 0:
            # TODO: load meshes
            self.meshes = []
            print(f"loaded {len(self.meshes)} meshes")
        else:
            self.meshes = []
        
        def get_poses_all(cameras):
            poses = []
            for camera in cameras:
                poses.append(camera.get_pose())
            poses = torch.stack(poses, 0)
            return poses

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

        # # bouding primitives
        # if bounding_primitive == "sphere":
        #     bouding_privimitive = Sphere()
        # else:
        #     bounding_primitive = AABB()

        # TODO: compute t_near, t_far by casting rays from all cameras,
        # intersecting with AABB [-1, 1] and returning min/max t
        # t_near, t_far = calculate_t_near_t_far(cameras_all, bouding_privimitive)

        # split data into train and test (or keep the all set)
        self.data = {}
        for split in splits:
            if split == "all":
                self.data[split] = cameras_all
            else:
                self.data[split] = []
                if split == "train":
                    if train_test_overlap:
                        # if train_test_overlap, use all cameras for training
                        self.data[split] = cameras_all
                    # else use only a subset of cameras
                    else:
                        for i, camera in enumerate(cameras_all):
                            if i % test_camera_freq != 0:
                                self.data[split].append(camera)
                if split == "test":
                    # select a test camera every test_camera_freq cameras
                    for i, camera in enumerate(cameras_all):
                        if i % test_camera_freq == 0:
                            self.data[split].append(camera)

            print(f"{split} split has {len(self.data[split])} cameras")

    def __getitem__(self, split):
        return self.data[split]


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


# def auto_orient_and_center_poses(
#     poses,
#     method="up",  # "up", "none"
#     center_method="focus",  # "poses", "focus", "none"
# ):
#     """Orients and centers the poses.

#     We provide three methods for orientation:

#     - up: Orient the poses so that the average up vector is aligned with the z axis.
#         This method works well when images are not at arbitrary angles.

#     There are two centering methods:

#     - poses: The poses are centered around the origin.
#     - focus: The origin is set to the focus of attention of all cameras (the
#         closest point to cameras optical axes). Recommended for inward-looking
#         camera configurations.

#     Args:
#         poses: The poses to orient.
#         method: The method to use for orientation.
#         center_method: The method to use to center the poses.

#     Returns:
#         Tuple of the oriented poses and the transform matrix.
#     """

#     origins = poses[..., :3, 3]

#     mean_origin = torch.mean(origins, dim=0)
#     # translation_diff = origins - mean_origin

#     if center_method == "poses":
#         translation = mean_origin
#     # elif center_method == "focus":
#     #     translation = focus_of_attention(poses, mean_origin)
#     elif center_method == "none":
#         translation = torch.zeros_like(mean_origin)
#     else:
#         raise ValueError(f"Unknown value for center_method: {center_method}")

#     if method == "up":
#         up = torch.mean(poses[:, :3, 1], dim=0)
#         up = up / torch.linalg.norm(up)
#         rotation = rotation_matrix(up.cpu().numpy(), np.array([0, 0, 1.0]))
#         rotation = torch.from_numpy(rotation)
#         # transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
#         transform = torch.eye(4)
#         transform[:3, :3] = rotation
#         transform[:3, 3] = -rotation @ -translation
#     elif method == "none":
#         transform = torch.eye(4)
#         transform[:3, 3] = -translation
#     else:
#         raise ValueError(f"Unknown value for method: {method}")

#     return transform
