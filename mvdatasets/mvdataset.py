from rich import print
from typing import List
import numpy as np
from pathlib import Path
from mvdatasets.utils.point_clouds import load_point_clouds
from mvdatasets.utils.printing import print_error, print_warning, print_info
from mvdatasets import Camera


DATASET_LOADER_MAPPING = {
    "nerf_synthetic": "blender",
    "nerf_furry": "blender",
    "refnerf": "blender",
    "shelly": "blender",
    "llff": "colmap",
    "mipnerf360": "colmap",
    "colmap": "colmap",
    "dtu": "dtu",
    "blended-mvs": "dtu",
    "dmsr": "dmsr",
    # "ingp": "ingp",
    "d-nerf": "d-nerf",
    "visor": "visor",
    "neu3d": "neu3d",
    "panoptic-sports": "panoptic-sports",
    "nerfies": "nerfies",
    "hypernerf": "nerfies",
    "iphone": "iphone",
    # preprocessing
    "monst3r": "monst3r",
    "iphone_som": "flow3d",
}


class MVDataset:
    """Any dataset container.
    All data is stored in CPU memory.
    """

    def __init__(
        self,
        dataset_name: str,
        scene_name: str,
        datasets_path: Path,
        config: dict,
        point_clouds_paths: list = [],
        verbose: bool = False,
    ):
        self.dataset_name = dataset_name
        self.scene_name = scene_name

        # datasets_path/dataset_name/scene_name
        dataset_path = Path(datasets_path) / dataset_name

        # check if path exists
        if not dataset_path.exists():
            raise ValueError(f"Data path {dataset_path} does not exist")

        print(f"dataset: [bold magenta]{dataset_name}[/bold magenta]")
        print(f"scene: [magenta]{scene_name}[/magenta]")
        print(f"loading {config['splits']} splits")

        self.cameras_on_hemisphere = False
        self.foreground_radius = 0.0
        self.init_sphere_radius = 0.0
        self.scene_radius = 0.0
        self.scene_type = ""
        self.global_transform = np.eye(4)
        self.nr_per_camera_frames = 0
        self.nr_sequence_frames = 0
        self.point_clouds = []
        self.fps = 0.0
        self.data = {}

        # STATIC SCENE DATASETS -----------------------------------------------

        loader = DATASET_LOADER_MAPPING.get(dataset_name, None)

        # dtu loader
        if loader == "dtu":
            from mvdatasets.loaders.static.dtu import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # blender loader
        elif loader == "blender":
            from mvdatasets.loaders.static.blender import load

            res = load(dataset_path, scene_name, config, verbose=verbose)
            self.cameras_on_hemisphere = True

        # ingp loader (deprecated)
        # elif loader == "ingp":
        #     res = load_ingp(dataset_path, scene_name, splits, config, verbose=verbose)

        # dmsr loader
        elif loader == "dmsr":
            from mvdatasets.loaders.static.dmsr import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # colmap loader
        elif loader == "colmap":
            from mvdatasets.loaders.static.colmap import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # DYNAMIC SCENE DATASETS ----------------------------------------------

        # d-nerf loader
        elif loader == "d-nerf":
            from mvdatasets.loaders.dynamic.d_nerf import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # visor loader
        elif loader == "visor":
            from mvdatasets.loaders.dynamic.visor import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # neu3d loader
        elif loader == "neu3d":
            from mvdatasets.loaders.dynamic.neu3d import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # panoptic-sports loader
        elif loader == "panoptic-sports":
            from mvdatasets.loaders.dynamic.panoptic_sports import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # nerfies loader
        elif loader == "nerfies":
            from mvdatasets.loaders.dynamic.nerfies import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # iphone loader
        elif loader == "iphone":
            from mvdatasets.loaders.dynamic.iphone import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # monst3r loader
        elif loader == "monst3r":
            from mvdatasets.loaders.dynamic.monst3r import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        # flow3d loader
        elif loader == "flow3d":
            from mvdatasets.loaders.dynamic.flow3d import load

            res = load(dataset_path, scene_name, config, verbose=verbose)

        else:

            raise ValueError(f"Dataset {dataset_name} is not supported")

        # UNPACK -------------------------------------------------------------

        if res is not None:

            # cameras
            cameras_splits = res["cameras_splits"]
            if cameras_splits is None or len(cameras_splits.keys()) == 0:
                raise ValueError("no cameras found")

            # config
            self.scene_type = res["scene_type"]
            self.global_transform = res["global_transform"]

            self.min_camera_distance = res["min_camera_distance"]
            self.max_camera_distance = res["max_camera_distance"]
            print("min_camera_distance:", self.min_camera_distance)
            print("max_camera_distance:", self.max_camera_distance)

            self.scene_radius = res["scene_radius"]
            print("scene_radius:", self.scene_radius)

            if "foreground_scale_mult" not in res:
                print_warning("foreground_scale_mult not found, setting to 0.5")
                self.foreground_scale_mult = 0.5
            else:
                self.foreground_scale_mult = res["foreground_scale_mult"]

            self.foreground_radius = self.scene_radius * self.foreground_scale_mult
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
                raise ValueError(
                    "init_sphere_radius > foreground_radius, this can't be true"
                )

            # dynamic scenes
            self.nr_per_camera_frames = res.get("nr_per_camera_frames", 1)
            print("nr_per_camera_frames:", self.nr_per_camera_frames)

            self.nr_sequence_frames = res.get("nr_sequence_frames", 1)
            print("nr_sequence_frames:", self.nr_sequence_frames)

            self.fps = res.get("fps", 0.0)
            print("fps:", self.fps)

            # optional
            self.point_clouds = res.get("point_clouds", [])
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
                    print_info(
                        f"{key} loaded with shape {val.shape}, dtype {val.dtype}"
                    )

    def get_split(self, split: str) -> List[Camera]:
        """Returns the list of cameras for a split"""
        if split not in self.get_splits():
            raise ValueError(
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

    def get_frame_rate(self) -> float:
        return self.fps

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
                raise ValueError(
                    f"camera index {camera_idx} out of range [0, {len(self.data[split])})"
                )
        else:
            raise ValueError(
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
                raise ValueError(
                    f"camera index {camera_idx} out of range [0, {len(self.data[split])})"
                )
        else:
            raise ValueError(
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
