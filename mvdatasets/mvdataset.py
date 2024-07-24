from rich import print
import os
import numpy as np
from mvdatasets.loaders.dtu import load_dtu
from mvdatasets.loaders.blender import load_blender
from mvdatasets.loaders.ingp import load_ingp
from mvdatasets.loaders.dmsr import load_dmsr
from mvdatasets.loaders.llff import load_llff
from mvdatasets.utils.point_clouds import load_point_clouds
from mvdatasets.utils.common import is_dataset_supported
from mvdatasets.utils.geometry import apply_transformation_3d
from mvdatasets.utils.contraction import contraction_function


def get_poses_all(cameras):
    poses = []
    for camera in cameras:
        poses.append(camera.get_pose())
    poses = np.stack(poses, 0)
    return poses


class MVDataset:
    """Dataset class for all static multi-view datasets.

    All data is stored in CPU memory.
    """

    def __init__(
        self,
        dataset_name,
        scene_name,
        datasets_path,
        splits=["train", "test"],  # ["train", "test"]
        point_clouds_paths=[],
        config={},  # if not specified, use default config
        # meshes_paths=[],
        # auto_orient_method="none",  # "up", "none"
        # auto_center_method="none",  # "poses", "focus", "none"
        # auto_scale_poses=False,  # scale the poses to fit in +/- 1 bounding box
        # profiler=None,
        verbose=False
    ):
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        # self.profiler = profiler

        # datasets_path/dataset_name/scene_name
        data_path = os.path.join(datasets_path, dataset_name, scene_name)

        # check if path exists
        if not os.path.exists(data_path):
            print(f"[bold red]ERROR[/bold red]: data path {data_path} does not exist")
            exit(1)

        # load scene cameras
        if splits is None:
            splits = ["all"]
        elif "train" not in splits and "test" not in splits:
            print("[bold red]ERROR[/bold red]: splits must contain at least one of 'train' or 'test'")
            exit(1)

        # check if dataset is supported
        if not is_dataset_supported(dataset_name):
            print(f"[bold red]ERROR[/bold red]: dataset {dataset_name} is not supported")
            exit(1)

        print(f"dataset: [bold magenta]{dataset_name}[/bold magenta]")
        print(f"scene: [magenta]{scene_name}[/magenta]")
        print(f"loading {splits} splits")
        
        self.cameras_on_hemisphere = False
        
        # STATIC SCENE DATASETS -----------------------------------------------
        
        # load dtu
        if self.dataset_name == "dtu":
            res = load_dtu(
                data_path,
                splits,
                config,
                verbose=verbose
            )

        # load blender
        # load blendernerf
        # load refnerf
        # load shelly
        elif (
                self.dataset_name == "blender"
                or self.dataset_name == "blendernerf"
                or self.dataset_name == "refnerf"
                or self.dataset_name == "shelly"
            ):
            res = load_blender(
                data_path,
                splits,
                config,
                verbose=verbose
            )
            self.cameras_on_hemisphere = True
        
        # load ingp
        elif self.dataset_name == "ingp":
            res = load_ingp(
                data_path,
                splits,
                config,
                verbose=verbose
            )
        
        # load dmsr
        elif self.dataset_name == "dmsr":
            res = load_dmsr(
                data_path,
                splits,
                config,
                verbose=verbose
            )
        
        # TODO: load multiface
        
        # TODO: load bmvs
        
        # load llff
        # load mipnerf360
        elif (
            self.dataset_name == "llff"
            or self.dataset_name == "mipnerf360"
        ):
            res = load_llff(
                data_path,
                splits,
                config,
                verbose=verbose
            )
        
        # TODO: load tanks_and_temples
        
        # TODO: ...
            
        # DYNAMIC SCENE DATASETS ----------------------------------------------

        # # load pac_nerf
        # elif self.dataset_name == "pac_nerf":
        #     # TODO: find n_cameras automatically
        #     cameras_splits = load_pac_nerf(
        #                                 data_path,
        #                                 splits,
        #                                 n_cameras=11,
        #                                 load_mask=load_mask
        #                             )

        # UNPACK -------------------------------------------------------------
    
        # cameras
        cameras_splits = res["cameras_splits"]
        
        # computed
        self.global_transform = res["global_transform"]
        self.min_camera_distance = res["min_camera_distance"]
        self.max_camera_distance = res["max_camera_distance"]
        self.scene_scale_mult = res["scene_scale_mult"]
        print("scene_scale:", res["scene_scale"])
        self.scene_radius = res["scene_scale"] * self.scene_scale_mult
        # round to 2 decimals
        self.scene_radius = round(self.scene_radius, 2)
        print("scene_radius:", self.scene_radius)
        
        # config
        self.scene_type = res["config"]["scene_type"]
        
        self.init_sphere_radius = (
            self.min_camera_distance \
            * self.scene_scale_mult \
            * res["config"]["init_sphere_scale"]
        )  # SDF sphere init radius
        # round to 2 decimals
        self.init_sphere_radius = round(self.init_sphere_radius, 2)
        print("init_sphere_radius:", self.init_sphere_radius)
        
        # optional
        if "point_clouds" in res:
            self.point_clouds = res["point_clouds"]
        else:
            self.point_clouds = []
        
        # ---------------------------------------------------------------------
        
        # (optional) load point clouds
        if len(self.point_clouds) == 0:
            # need to load point clouds
            if len(point_clouds_paths) > 0:
                # load point clouds
                self.point_clouds = load_point_clouds(point_clouds_paths, verbose=verbose)
                if verbose:
                    print(f"loaded {len(self.point_clouds)} point clouds")
        else:
            if len(point_clouds_paths) > 0:
                print("[bold yellow]WARNING[/bold yellow]: point_clouds_paths will be ignored")
        
        transformed_point_clouds = []
        for point_cloud in self.point_clouds:
            # apply global transform
            pc = apply_transformation_3d(point_cloud, self.global_transform)
            # apply contraction function
            if self.scene_type == "unbounded":
                pc = contraction_function(pc)
            transformed_point_clouds.append(pc)
        self.point_clouds = transformed_point_clouds
        
        # TODO: (optional) load meshes
        # if len(meshes_paths) > 0:
        #     # TODO: load meshes
        #     self.meshes = []
        #     print(f"loaded {len(self.meshes)} meshes")
        # else:
        #     self.meshes = []

        # # align and center poses
        cameras_all = []
        for split, cameras_list in cameras_splits.items():
            cameras_all += cameras_list
        
        # transform = auto_orient_and_center_poses(
        #     poses_all,
        #     method=auto_orient_method,
        #     center_method=auto_center_method,
        # )

        # for camera in cameras_all:
        #     camera.concat_global_transform(transform)

        # # scale poses
        # if self.auto_scale_poses:
        #     scale_factor = float(
        #         torch.max(torch.abs(get_poses_all(cameras_all)[:, :3, 3]))
        #     )
        #     transform = torch.eye(4)
        #     transform[3, 3] = scale_factor
        #     for camera in cameras_all:
        #         camera.concat_global_transform(transform)

        # # bouding primitives
        # if bounding_primitive == "sphere":
        #     bouding_privimitive = Sphere()
        # else:
        #     bounding_primitive = AABB()

        # TODO: compute t_near, t_far by casting rays from all cameras,
        # intersecting with AABB [-1, 1] and returning min/max t
        # t_near, t_far = calculate_t_near_t_far(cameras_all, bouding_privimitive)

        # split data into train and test (or keep the all set)
        self.data = cameras_splits
        
        for split in splits:
            print(f"{split} split has {len(self.data[split])} cameras")
                    
    def has_masks(self):
        for split, cameras in self.data.items():
            for camera in cameras:
                # assumption: if one camera has masks, all cameras have masks
                if camera.has_masks():
                    return True
        return False
    
    def get_width(self, split="train", camera_id=0):
        """Returns the width of a camera

        Args:
            split (str, optional): Defaults to "train".
            camera_id (int, optional): Defaults to 0.

        Returns:
            int: width
        """
        if split in self.data:
            if camera_id >= 0 and camera_id < len(self.data[split]):
                return self.data[split][camera_id].width
            else:
                print(f"[bold red]ERROR[/bold red]: camera index {camera_id} out of range [0, {len(self.data[split])})")
                exit(1)
        else:
            print(f"[bold red]ERROR[/bold red]: split {split} does not exist, available splits: {list(self.data.keys())}")
            exit(1)
            
    def get_height(self, split="train", camera_id=0):
        """Returns the height of a camera

        Args:
            split (str, optional): Defaults to "train".
            camera_id (int, optional): Defaults to 0.

        Returns:
            int: height
        """
        if split in self.data:
            if camera_id >= 0 and camera_id < len(self.data[split]):
                return self.data[split][camera_id].height
            else:
                print(f"[bold red]ERROR[/bold red]: camera index {camera_id} out of range [0, {len(self.data[split])})")
                exit(1)
        else:
            print(f"[bold red]ERROR[/bold red]: split {split} does not exist, available splits: {list(self.data.keys())}")
            exit(1)
            
    def get_resolution(self, split="train", camera_id=0):
        """Returns the resolution (width, height) of a camera

        Args:
            split (str, optional): Defaults to "train".
            camera_id (int, optional): Defaults to 0.

        Returns:
            (int, int): width, height
        """
        return (self.get_width(split, camera_id), self.get_height(split, camera_id))
                    
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
