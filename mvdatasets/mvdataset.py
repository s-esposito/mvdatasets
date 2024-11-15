from rich import print
import os
import numpy as np
from mvdatasets.loaders.dtu import load_dtu
from mvdatasets.loaders.blender import load_blender
from mvdatasets.loaders.ingp import load_ingp
from mvdatasets.loaders.dmsr import load_dmsr
from mvdatasets.loaders.llff import load_llff
from mvdatasets.utils.point_clouds import load_point_clouds
from mvdatasets.config import is_dataset_supported
from mvdatasets.utils.geometry import apply_transformation_3d
from mvdatasets.utils.contraction import contract_points
from mvdatasets.utils.printing import print_error, print_warning


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
        pose_only=False,  # if set, does not load images
        verbose=False
    ):
        self.dataset_name = dataset_name
        self.scene_name = scene_name
        # self.profiler = profiler
        
        config["pose_only"] = pose_only

        # datasets_path/dataset_name/scene_name
        data_path = os.path.join(datasets_path, dataset_name, scene_name)

        # check if path exists
        if not os.path.exists(data_path):
            print_error(f"data path {data_path} does not exist")

        # load scene cameras
        if splits is None:
            splits = ["all"]
        elif "train" not in splits and "test" not in splits:
            print_error("splits must contain at least one of 'train' or 'test'")

        # check if dataset is supported
        if not is_dataset_supported(dataset_name):
            print_error(f"dataset {dataset_name} is not supported")

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
        
        # SDF sphere init radius
        # for SDF reconstruction
        self.init_sphere_radius = (
            self.min_camera_distance
            * self.scene_scale_mult
            * res["config"]["init_sphere_scale"]
        )  
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
                print_warning("point_clouds_paths will be ignored")
        
        transformed_point_clouds = []
        for point_cloud in self.point_clouds:
            # apply global transform
            pc = apply_transformation_3d(point_cloud, self.global_transform)
            # apply contraction function
            # if self.scene_type == "unbounded":
            #     pc = contract_points(pc)
            transformed_point_clouds.append(pc)
        self.point_clouds = transformed_point_clouds

        # split data into train and test (or keep the all set)
        self.data = cameras_splits
        
        # printing
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
                print_error(f"camera index {camera_id} out of range [0, {len(self.data[split])})")
        else:
            print_error(f"split {split} does not exist, available splits: {list(self.data.keys())}")
            
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
                print_error(f"camera index {camera_id} out of range [0, {len(self.data[split])})")
        else:
            print_error(f"split {split} does not exist, available splits: {list(self.data.keys())}")
            
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