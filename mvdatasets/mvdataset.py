from rich import print
import os
from typing import List, Dict, Any
import numpy as np
from pathlib import Path
from mvdatasets.utils.point_clouds import load_point_clouds
from mvdatasets.utils.printing import print_error, print_warning, print_info
from mvdatasets import Camera
from mvdatasets.loaders.configs import get_scene_preset


class MVDataset:
    """Dataset class for all static multi-view datasets.

    All data is stored in CPU memory.
    """

    def __init__(
        self,
        dataset_name: str,
        scene_name: str,
        datasets_path: Path,
        splits: list,
        point_clouds_paths: list = [],
        config: dict = {},  # if not specified, use default config
        pose_only: bool = False,  # if set, does not load images
        verbose: bool = False,
    ):
        self.dataset_name = dataset_name
        self.scene_name = scene_name

        # load dataset and scene default config
        default_config = get_scene_preset(dataset_name, scene_name)

        # override default config with user config
        for key, value in config.items():
            default_config[key] = value
        config = default_config

        # default config
        config["pose_only"] = pose_only

        # datasets_path/dataset_name/scene_name
        dataset_path = Path(datasets_path) / dataset_name

        # check if path exists
        if not dataset_path.exists():
            print_error(f"data path {dataset_path} does not exist")

        print(f"dataset: [bold magenta]{dataset_name}[/bold magenta]")
        print(f"scene: [magenta]{scene_name}[/magenta]")
        print(f"loading {splits} splits")

        self.cameras_on_hemisphere = False

        # STATIC SCENE DATASETS -----------------------------------------------

        # load dtu
        if self.dataset_name == "dtu" or self.dataset_name == "blended-mvs":
            from mvdatasets.loaders.static.dtu import load

            res = load(dataset_path, scene_name, splits, config, verbose=verbose)

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

            res = load(dataset_path, scene_name, splits, config, verbose=verbose)
            self.cameras_on_hemisphere = True

        # # load ingp
        # elif self.dataset_name == "ingp":
        #     res = load_ingp(dataset_path, scene_name, splits, config, verbose=verbose)

        # load dmsr
        elif self.dataset_name == "dmsr":
            from mvdatasets.loaders.static.dmsr import load

            res = load(dataset_path, scene_name, splits, config, verbose=verbose)

        # load generic colmap reconstruction
        # load llff
        # load mipnerf360
        elif (
            self.dataset_name == "colmap"
            or self.dataset_name == "llff"
            or self.dataset_name == "mipnerf360"
        ):
            from mvdatasets.loaders.static.colmap import load

            res = load(dataset_path, scene_name, splits, config, verbose=verbose)

        # DYNAMIC SCENE DATASETS ----------------------------------------------

        elif self.dataset_name == "d-nerf":
            from mvdatasets.loaders.dynamic.d_nerf import load

            res = load(dataset_path, scene_name, splits, config, verbose=verbose)

        elif self.dataset_name == "visor":
            from mvdatasets.loaders.dynamic.visor import load

            res = load(dataset_path, scene_name, splits, config, verbose=verbose)

        elif self.dataset_name == "panoptic-sports":
            from mvdatasets.loaders.dynamic.panoptic_sports import load

            res = load(dataset_path, scene_name, splits, config, verbose=verbose)

        elif self.dataset_name == "nerfies":
            from mvdatasets.loaders.dynamic.nerfies import load

            res = load(dataset_path, scene_name, splits, config, verbose=verbose)

        # elif self.dataset_name == "iphone":
        #     from mvdatasets.loaders.dynamic.iphone import load

        #     res = load(dataset_path, scene_name, splits, config, verbose=verbose)

        else:
            
            print_warning(f"dataset {self.dataset_name} is not supported")
            res = None
        
        # UNPACK -------------------------------------------------------------

        if res is not None:
        
            # cameras
            cameras_splits = res["cameras_splits"]
            if cameras_splits is None or len(cameras_splits.keys()) == 0:
                print_error("no cameras found")  # this should never happen

            # config
            self.scene_type = res["scene_type"]
            self.global_transform = res["global_transform"]

            self.min_camera_distance = res["min_camera_distance"]
            self.max_camera_distance = res["max_camera_distance"]
            print("min_camera_distance:", self.min_camera_distance)
            print("max_camera_distance:", self.max_camera_distance)

            self.scene_radius = res["scene_radius"]
            print("scene_radius:", self.scene_radius)

            if "foreground_radius_mult" not in res:
                print_warning("foreground_radius_mult not found, setting to 0.5")
                self.foreground_radius_mult = 0.5
            else:
                self.foreground_radius_mult = res["foreground_radius_mult"]

            self.foreground_radius = self.scene_radius * self.foreground_radius_mult
            print("foreground_radius:", self.foreground_radius)

            # SDF sphere init radius
            if "init_sphere_radius_mult" not in res:
                print_warning("init_sphere_radius_mult not found, setting to 0.1")
                self.init_sphere_radius_mult = 0.1
            else:
                self.init_sphere_radius_mult = res["init_sphere_radius_mult"]

            self.init_sphere_radius = (
                self.min_camera_distance * self.init_sphere_radius_mult
            )
            print("init_sphere_radius:", self.init_sphere_radius)

            if self.init_sphere_radius > self.foreground_radius:
                print_error("init_sphere_radius > scene_radius, this can't be true")

            # dynamic scenes
            if "nr_per_camera_frames" in res:
                self.nr_per_camera_frames = res["nr_per_camera_frames"]
            else:
                self.nr_per_camera_frames = 1
            print("nr_per_camera_frames:", self.nr_per_camera_frames)

            if "nr_sequence_frames" in res:
                self.nr_sequence_frames = res["nr_sequence_frames"]
            else:
                self.nr_sequence_frames = 1
            print("nr_sequence_frames:", self.nr_sequence_frames)

            # optional
            if "point_clouds" in res:
                self.point_clouds = res["point_clouds"]
            else:
                self.point_clouds = []
            print("loaded scene has", len(self.point_clouds), "point clouds")

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

            for point_cloud in self.point_clouds:
                # apply global transform
                point_cloud.transform(self.global_transform)

            # split data into train and test (or keep the all set)
            self.data = cameras_splits
            
        else:
            # res is None
            self.data = {}
            # TODO: better handling all other attributes when dataset is not supported
        
        # printing
        for split in self.data.keys():
            print_fn = print_info
            if len(self.data[split]) == 0:
                print_fn = print_error
            print_fn(f"{split} split has {len(self.data[split])} cameras")
            # print modalities loaded
            for key, val in self.data[split][0].data.items():
                if val is not None:
                    print_info(f"{key} loaded with shape {val.shape}")

    def get_split(self, split: str) -> List[Camera]:
        """Returns the list of cameras for a split"""
        if split not in self.get_splits():
            print_error(
                f"split {split} does not exist, available splits: {list(self.data.keys())}"
            )
        return self.data[split]

    def get_splits(self) -> List[str]:
        """Returns the list of splits"""
        return list(self.data.keys())

    def get_split_modalities(self, split: str) -> List[str]:
        """Returns the list of modalities for a split"""
        cameras = self.get_split(split)
        return cameras[0].get_available_data()

    def get_sphere_init_radius(self) -> float:
        return self.init_sphere_radius

    def get_scene_type(self) -> str:
        return self.scene_type

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

    def get_nr_per_camera_frames(self) -> int:
        """Returns the sequence length of the dataset"""
        return self.nr_per_camera_frames

    def get_nr_sequence_frames(self) -> int:
        """Returns the sequence length of the dataset"""
        return self.nr_sequence_frames

    def get_width(self, split: str = "train", camera_idx: int = 0) -> int:
        """Returns the width of a camera

        Args:
            split (str, optional): Defaults to "train".
            camera_idx (int, optional): Defaults to 0.

        Returns:
            int: width
        """
        if split in self.data:
            if camera_idx >= 0 and camera_idx < len(self.data[split]):
                return self.data[split][camera_idx].width
            else:
                print_error(
                    f"camera index {camera_idx} out of range [0, {len(self.data[split])})"
                )
        else:
            print_error(
                f"split {split} does not exist, available splits: {list(self.data.keys())}"
            )

    def get_height(self, split: str = "train", camera_idx: int = 0) -> int:
        """Returns the height of a camera

        Args:
            split (str, optional): Defaults to "train".
            camera_idx (int, optional): Defaults to 0.

        Returns:
            int: height
        """
        if split in self.data:
            if camera_idx >= 0 and camera_idx < len(self.data[split]):
                return self.data[split][camera_idx].height
            else:
                print_error(
                    f"camera index {camera_idx} out of range [0, {len(self.data[split])})"
                )
        else:
            print_error(
                f"split {split} does not exist, available splits: {list(self.data.keys())}"
            )

    def get_resolution(self, split="train", camera_idx=0) -> tuple:
        """Returns the resolution (width, height) of a camera

        Args:
            split (str, optional): Defaults to "train".
            camera_idx (int, optional): Defaults to 0.

        Returns:
            (int, int): width, height
        """
        return (self.get_width(split, camera_idx), self.get_height(split, camera_idx))

    def __getitem__(self, split: str) -> List[Camera]:
        """Returns the list of cameras for a split"""
        return self.data[split]
