from rich import print
import os
from typing import List
import numpy as np
from pathlib import Path
from mvdatasets.utils.point_clouds import load_point_clouds
from mvdatasets.geometry.common import apply_transformation_3d
from mvdatasets.utils.printing import print_error, print_warning
from mvdatasets import Camera


class MVDataset:
    """Dataset class for all static multi-view datasets.

    All data is stored in CPU memory.
    """

    def __init__(
        self,
        dataset_name: str,
        scene_name: str,
        datasets_path: Path,
        splits: list = ["train", "test"],
        point_clouds_paths: list = [],
        config: dict = {},  # if not specified, use default config
        pose_only: bool = False,  # if set, does not load images
        verbose: bool = False,
    ):
        self.dataset_name = dataset_name
        self.scene_name = scene_name

        # default config
        config["pose_only"] = pose_only

        # datasets_path/dataset_name/scene_name
        data_path = Path(datasets_path) / dataset_name / scene_name

        # check if path exists
        if not data_path.exists():
            print_error(f"data path {data_path} does not exist")

        # load scene cameras
        if splits is None:
            splits = ["all"]
        elif "train" not in splits and "test" not in splits:
            print_error("splits must contain at least one of 'train' or 'test'")

        print(f"dataset: [bold magenta]{dataset_name}[/bold magenta]")
        print(f"scene: [magenta]{scene_name}[/magenta]")
        print(f"loading {splits} splits")

        self.cameras_on_hemisphere = False

        # STATIC SCENE DATASETS -----------------------------------------------

        # load dtu
        if self.dataset_name == "dtu" or self.dataset_name == "blended-mvs":
            from mvdatasets.loaders.static.dtu import load

            res = load(data_path, splits, config, verbose=verbose)

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
            from mvdatasets.loaders.static.blender import load

            res = load(data_path, splits, config, verbose=verbose)
            self.cameras_on_hemisphere = True

        # # load ingp
        # elif self.dataset_name == "ingp":
        #     res = load_ingp(data_path, splits, config, verbose=verbose)

        # load dmsr
        elif self.dataset_name == "dmsr":
            from mvdatasets.loaders.static.dmsr import load

            res = load(data_path, splits, config, verbose=verbose)

        # load llff
        # load mipnerf360
        elif self.dataset_name == "llff" or self.dataset_name == "mipnerf360":
            from mvdatasets.loaders.static.colmap import load

            res = load(data_path, splits, config, verbose=verbose)

        # DYNAMIC SCENE DATASETS ----------------------------------------------

        elif self.dataset_name == "d-nerf":
            from mvdatasets.loaders.dynamic.d_nerf import load

            res = load(data_path, splits, config, verbose=verbose)

        elif self.dataset_name == "panoptic-sports":
            from mvdatasets.loaders.dynamic.panoptic_sports import load

            res = load(data_path, splits, config, verbose=verbose)

        elif self.dataset_name == "nerfies":
            from mvdatasets.loaders.dynamic.nerfies import load

            res = load(data_path, splits, config, verbose=verbose)

        elif self.dataset_name == "iphone":
            from mvdatasets.loaders.dynamic.iphone import load

            res = load(data_path, splits, config, verbose=verbose)

        # UNPACK -------------------------------------------------------------

        else:
            print_error(f"dataset {self.dataset_name} is not supported")

        # cameras
        cameras_splits = res["cameras_splits"]

        # config
        self.scene_type = res["scene_type"]
        self.global_transform = res["global_transform"]
        
        self.min_camera_distance = res["min_camera_distance"]
        self.max_camera_distance = res["max_camera_distance"]
        print("min_camera_distance:", self.min_camera_distance)
        print("max_camera_distance:", self.max_camera_distance)
        
        self.scene_radius = res["scene_radius"]
        self.foreground_radius_mult = res["foreground_radius_mult"]
        self.foreground_radius = self.scene_radius * self.foreground_radius_mult
        print("scene_radius:", self.scene_radius)
        print("foreground_radius:", self.foreground_radius)

        # SDF sphere init radius
        # for SDF reconstruction
        self.init_sphere_radius = (
            self.min_camera_distance
            * res["init_sphere_scale"]
        )
        # round to 2 decimals
        self.init_sphere_radius = round(self.init_sphere_radius, 2)
        print("init_sphere_radius:", self.init_sphere_radius)
        
        if self.init_sphere_radius > self.foreground_radius:
            print_error("init_sphere_radius > scene_radius, this can't be true")

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
                self.point_clouds = load_point_clouds(
                    point_clouds_paths, verbose=verbose
                )
                if verbose:
                    print(f"loaded {len(self.point_clouds)} point clouds")
        else:
            if len(point_clouds_paths) > 0:
                print_warning("point_clouds_paths will be ignored")

        transformed_point_clouds = []
        for point_cloud in self.point_clouds:
            # apply global transform
            pc = apply_transformation_3d(point_cloud, self.global_transform)
            transformed_point_clouds.append(pc)
        self.point_clouds = transformed_point_clouds

        # split data into train and test (or keep the all set)
        self.data = cameras_splits

        # printing
        for split in splits:
            print(f"{split} split has {len(self.data[split])} cameras")

    def get_sphere_init_radius(self) -> float:
        return self.init_sphere_radius
    
    def get_scene_radius(self) -> float:
        return self.scene_radius
    
    def get_foreground_radius(self) -> float:
        return self.foreground_radius
    
    def has_masks(self) -> bool:
        for split, cameras in self.data.items():
            for camera in cameras:
                # assumption: if one camera has masks, all cameras have masks
                if camera.has_masks():
                    return True
        return False

    def get_width(self, split: str = "train", camera_id: int = 0) -> int:
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
                print_error(
                    f"camera index {camera_id} out of range [0, {len(self.data[split])})"
                )
        else:
            print_error(
                f"split {split} does not exist, available splits: {list(self.data.keys())}"
            )

    def get_height(self, split: str = "train", camera_id: int = 0) -> int:
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
                print_error(
                    f"camera index {camera_id} out of range [0, {len(self.data[split])})"
                )
        else:
            print_error(
                f"split {split} does not exist, available splits: {list(self.data.keys())}"
            )

    def get_resolution(self, split="train", camera_id=0) -> tuple:
        """Returns the resolution (width, height) of a camera

        Args:
            split (str, optional): Defaults to "train".
            camera_id (int, optional): Defaults to 0.

        Returns:
            (int, int): width, height
        """
        return (self.get_width(split, camera_id), self.get_height(split, camera_id))

    def __getitem__(self, split: str) -> List[Camera]:
        """Returns the list of cameras for a split"""
        return self.data[split]
