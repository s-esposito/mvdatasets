from collections import OrderedDict
from typing import Dict
import tyro
from dataclasses import dataclass
from mvdatasets.configs.dataset_config import DatasetConfig

# Static Datasets Configurations ------------------------------------------------


@dataclass
class BlenderConfig(DatasetConfig):
    # Default dataset configuration

    load_masks: bool = True
    """Load mask images"""
    use_binary_mask: bool = True
    """Convert masks to binary"""
    white_bg: bool = True
    """Use white background (else black)"""
    test_skip: int = 20
    """Skip every test_skip images from test split"""

    def __post__init__(self):
        #
        super().__post__init__()
        # load_masks
        if type(self.load_masks) is not bool:
            raise ValueError("load_masks must be a boolean")
        # use_binary_mask
        if type(self.use_binary_mask) is not bool:
            raise ValueError("use_binary_mask must be a boolean")
        # white_bg
        if type(self.white_bg) is not bool:
            raise ValueError("white_bg must be a boolean")
        # test_skip
        if type(self.test_skip) is not int or self.test_skip < 1:
            raise ValueError("test_skip must be an integer >= 1")
        # check if splits are valid
        valid_splits = ["train", "test"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class DTUConfig(DatasetConfig):
    # Default dataset configuration

    load_masks: bool = False
    """Load mask images"""
    test_camera_freq: int = 8
    """Sample a camera every test_camera_freq cameras from all cameras for test split"""
    train_test_overlap: bool = False
    """Use all cameras for training if True, else use a subset of cameras"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # load_masks
        if type(self.load_masks) is not bool:
            raise ValueError("load_masks must be a boolean")
        # test_camera_freq
        if type(self.test_camera_freq) is not int or self.test_camera_freq < 1:
            raise ValueError("test_camera_freq must be an integer >= 1")
        # train_test_overlap
        if type(self.train_test_overlap) is not bool:
            raise ValueError("train_test_overlap must be a boolean")
        # check if splits are valid
        valid_splits = ["train", "test"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class ColmapConfig(DatasetConfig):
    # Default dataset configuration

    test_camera_freq: int = 8
    """Sample a camera every test_camera_freq cameras from all cameras for test split"""
    train_test_overlap: bool = False
    """Use all cameras for training if True, else use a subset of cameras"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # test_camera_freq
        if type(self.test_camera_freq) is not int or self.test_camera_freq < 1:
            raise ValueError("test_camera_freq must be an integer >= 1")
        # train_test_overlap
        if type(self.train_test_overlap) is not bool:
            raise ValueError("train_test_overlap must be a boolean")
        # check if splits are valid
        valid_splits = ["train", "test"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class DMRSConfig(DatasetConfig):
    # Default dataset configuration

    test_skip: int = 20
    """Skip every test_skip images from test split"""

    # load_depth: bool = False  # Load depth images
    # load_semantics: bool = False  # Load semantic images
    # load_semantic_instance: bool = False  # Load semantic instance images

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # test_skip
        if type(self.test_skip) is not int or self.test_skip < 1:
            raise ValueError("test_skip must be an integer >= 1")
        # # load_depth
        # if type(self.load_depth) is not bool:
        #     raise ValueError("load_depth must be a boolean")
        # # load_semantics
        # if type(self.load_semantics) is not bool:
        #     raise ValueError("load_semantics must be a boolean")
        # # load_semantic_instance
        # if type(self.load_semantic_instance) is not bool:
        #     raise ValueError("load_semantic_instance must be a boolean")
        # check if splits are valid
        valid_splits = ["train", "test"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


# Dynamic Datasets Configurations -----------------------------------------------


@dataclass
class DNeRFConfig(DatasetConfig):
    # Default dataset configuration

    load_masks: bool = True
    """Load mask images"""
    use_binary_mask: bool = True
    """Convert masks to binary"""
    white_bg: bool = True
    """Use white background (else black)"""
    frame_rate: float = 30.0
    """Frame rate of the sequence"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # load_masks
        if type(self.load_masks) is not bool:
            raise ValueError("load_masks must be a boolean")
        # use_binary_mask
        if type(self.use_binary_mask) is not bool:
            raise ValueError("use_binary_mask must be a boolean")
        # white_bg
        if type(self.white_bg) is not bool:
            raise ValueError("white_bg must be a boolean")
        # frame_rate
        if type(self.frame_rate) is not float or self.frame_rate <= 0:
            raise ValueError("frame_rate must be a float > 0")
        # check if splits are valid
        valid_splits = ["train", "test"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class VISORConfig(DatasetConfig):
    # Default dataset configuration

    load_masks: bool = True
    """Load mask images"""
    load_semantic_masks: bool = True
    """Load semantic mask images"""
    frame_rate: float = 10.0
    """Frame rate of the sequence"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # load_masks
        if type(self.load_masks) is not bool:
            raise ValueError("load_masks must be a boolean")
        # load_semantic_masks
        if type(self.load_semantic_masks) is not bool:
            raise ValueError("load_semantic_masks must be a boolean")
        # frame_rate
        if type(self.frame_rate) is not float or self.frame_rate <= 0:
            raise ValueError("frame_rate must be a float > 0")
        # check if splits are valid
        valid_splits = ["train", "val"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class Neu3DConfig(DatasetConfig):
    # Default dataset configuration

    frame_rate: float = 30.0
    """Frame rate of the sequence"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # frame_rate
        if type(self.frame_rate) is not float or self.frame_rate <= 0:
            raise ValueError("frame_rate must be a float > 0")
        # check if splits are valid
        valid_splits = ["train", "test"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class PanopticSportsConfig(DatasetConfig):
    # Default dataset configuration

    load_masks: bool = True
    """Load mask images"""
    frame_rate: float = 30.0
    """Frame rate of the sequence"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # load_masks
        if type(self.load_masks) is not bool:
            raise ValueError("load_masks must be a boolean")
        # frame_rate
        if type(self.frame_rate) is not float or self.frame_rate <= 0:
            raise ValueError("frame_rate must be a float > 0")
        # check if splits are valid
        valid_splits = ["train", "test"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class NerfiesConfig(DatasetConfig):
    # Default dataset configuration

    # load_masks: bool = True
    # """Load mask images"""
    load_depths: bool = True
    """Load depth images"""
    frame_rate: float = 30.0
    """Frame rate of the sequence"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # load_masks
        # if type(self.load_masks) is not bool:
        #     raise ValueError("load_masks must be a boolean")
        # load_depths
        if type(self.load_depths) is not bool:
            raise ValueError("load_depths must be a boolean")
        # frame_rate
        if type(self.frame_rate) is not float or self.frame_rate <= 0:
            raise ValueError("frame_rate must be a float > 0")
        # check if splits are valid
        valid_splits = ["train", "val"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class iPhoneConfig(DatasetConfig):
    # Default dataset configuration

    # load_masks: bool = True
    # """Load mask images"""
    load_depths: bool = True
    """Load depth images"""
    frame_rate: float = 30.0
    """Frame rate of the sequence"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # load_masks
        # if type(self.load_masks) is not bool:
        #     raise ValueError("load_masks must be a boolean")
        # load_depths
        if type(self.load_depths) is not bool:
            raise ValueError("load_depths must be a boolean")
        # frame_rate
        if type(self.frame_rate) is not float or self.frame_rate <= 0:
            raise ValueError("frame_rate must be a float > 0")
        # check if splits are valid
        valid_splits = ["train", "val"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class MonST3RConfig(DatasetConfig):
    # Default dataset configuration

    load_masks: bool = True
    """Load mask images"""
    load_depths: bool = True
    """Load depth images"""
    frame_rate: float = 30.0
    """Frame rate of the sequence"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # load_masks
        if type(self.load_masks) is not bool:
            raise ValueError("load_masks must be a boolean")
        # load_depths
        if type(self.load_depths) is not bool:
            raise ValueError("load_depths must be a boolean")
        # frame_rate
        if type(self.frame_rate) is not float or self.frame_rate <= 0:
            raise ValueError("frame_rate must be a float > 0")
        # check if splits are valid
        valid_splits = ["train"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


@dataclass
class Flow3DConfig(DatasetConfig):
    # Default dataset configuration

    load_masks: bool = True
    """Load mask images"""
    load_depths: bool = True
    """Load depth images"""
    load_2d_tracks: bool = True
    """Load 2D tracks"""
    load_3d_tracks: bool = True
    """Load 3D tracks"""
    frame_rate: float = 30.0
    """Frame rate of the sequence"""

    def __post__init__(self):
        # Check configuration values
        super().__post__init__()
        # load_masks
        if type(self.load_masks) is not bool:
            raise ValueError("load_masks must be a boolean")
        # load_depths
        if type(self.load_depths) is not bool:
            raise ValueError("load_depths must be a boolean")
        # load_2d_tracks
        if type(self.load_2d_tracks) is not bool:
            raise ValueError("load_2d_tracks must be a boolean")
        # load_3d_tracks
        if type(self.load_3d_tracks) is not bool:
            raise ValueError("load_3d_tracks must be a boolean")
        # frame_rate
        if type(self.frame_rate) is not float or self.frame_rate <= 0:
            raise ValueError("frame_rate must be a float > 0")
        # check if splits are valid
        valid_splits = ["train", "val"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"split {split} not supported, must be one of {valid_splits}"
                )


# -------------------------------------------------------------------------------


# Registry for dataset configurations
datasets_configs: Dict[str, DatasetConfig] = {
    # Blender format
    "nerf_synthetic": BlenderConfig(
        dataset_name="nerf_synthetic",
        splits=["train", "test"],
        scene_type="bounded",
        test_skip=20,
    ),
    "nerf_furry": BlenderConfig(
        dataset_name="nerf_furry",
        splits=["train", "test"],
        scene_type="bounded",
        test_skip=10,
    ),
    "shelly": BlenderConfig(
        dataset_name="shelly",
        splits=["train", "test"],
        scene_type="bounded",
        test_skip=4,
    ),
    "refnerf": BlenderConfig(
        dataset_name="refnerf",
        splits=["train", "test"],
        scene_type="bounded",
        test_skip=10,
    ),
    # DTU format
    "dtu": DTUConfig(
        dataset_name="dtu",
        splits=["train", "test"],
        scene_type="unbounded",
        load_masks=True,
        rotate_deg=[205.0, 0.0, 0.0],
        max_cameras_distance=0.5,
        foreground_scale_mult=1.0,
    ),
    "blended-mvs": DTUConfig(
        dataset_name="blended-mvs",
        splits=["train", "test"],
        scene_type="unbounded",
        load_masks=True,
        rotate_deg=[0.0, 0.0, 0.0],
        max_cameras_distance=0.5,
        foreground_scale_mult=1.0,
    ),
    # DMRS format
    "dmsr": DMRSConfig(
        dataset_name="dmsr",
        splits=["train", "test"],
        scene_type="ubounded",
        test_skip=5,
        foreground_scale_mult=0.5,
        max_cameras_distance=1.0,
        init_sphere_radius_mult=0.3,
        # load_depth=False,
        # load_semantics=False,
        # load_semantic_instance=False,
    ),
    # INGP format
    # "ingp": INGPConfig(
    #     dataset_name="ingp",
    #     splits=["train", "test"],
    #     scene_type="bounded",
    # ),
    # COLMAP format
    "llff": ColmapConfig(
        dataset_name="llff",
        splits=["train", "test"],
        scene_type="unbounded",
    ),
    "mipnerf360": ColmapConfig(
        dataset_name="mipnerf360",
        splits=["train", "test"],
        scene_type="unbounded",
        subsample_factor=8,
        foreground_scale_mult=0.5,
    ),
    # D-Nerf format
    "d-nerf": DNeRFConfig(
        dataset_name="d-nerf",
        splits=["train", "test"],
        scene_type="bounded",
        load_masks=True,
        use_binary_mask=True,
        white_bg=True,
        max_cameras_distance=None,
        foreground_scale_mult=0.5,
        init_sphere_radius_mult=0.3,
        frame_rate=30.0,
    ),
    # VISOR format
    "visor": VISORConfig(
        dataset_name="visor",
        splits=["train"],
        scene_type="unbounded",
        rotate_deg=[180.0, 0.0, 0.0],
        load_masks=True,
        load_semantic_masks=True,
        max_cameras_distance=None,  # no scaling
        foreground_scale_mult=1.0,
        frame_rate=10.0,
    ),
    # Neu3D format
    "neu3d": Neu3DConfig(
        dataset_name="neu3d",
        splits=["train", "test"],
        scene_type="unbounded",
    ),
    # Panoptic-Sports format
    "panoptic-sports": PanopticSportsConfig(
        dataset_name="panoptic-sports",
        splits=["train", "test"],
        scene_type="unbounded",
    ),
    # Nerfies format
    "nerfies": NerfiesConfig(
        dataset_name="nerfies",
        splits=["train", "val"],
        scene_type="unbounded",
        subsample_factor=2,
        load_depths=False,
    ),
    "iphone": iPhoneConfig(
        dataset_name="iphone",
        splits=["train", "val"],
        scene_type="unbounded",
        subsample_factor=2,
        load_depths=True,
        rotate_deg=[90.0, 0.0, 0.0],
        max_cameras_distance=1.0,  # scale to 1.0
    ),
    # MonST3R format
    "monst3r": MonST3RConfig(
        dataset_name="monst3r",
        splits=["train"],
        scene_type="unbounded",
        subsample_factor=1,
        load_masks=True,
        load_depths=True,
        rotate_deg=[90.0, 0.0, 0.0],
        max_cameras_distance=1.0,  # scale to 1.0
    ),
    # Flow3D format
    "iphone_som": Flow3DConfig(
        dataset_name="iphone_som",
        splits=["train", "val"],
        scene_type="unbounded",
        subsample_factor=2,
        load_masks=True,
        load_depths=True,
        load_2d_tracks=True,
        load_3d_tracks=True,
        rotate_deg=[90.0, 0.0, 0.0],
        max_cameras_distance=None,  # no scaling
    ),
}

datasets_descriptions: Dict[str, str] = {
    "nerf_synthetic": "NeRF Synthetic dataset",
    "nerf_furry": "NeRF Furry dataset",
    "dtu": "DTU dataset",
    "blended-mvs": "BlendedMVS dataset",
    "dmsr": "DMSR dataset",
    "llff": "LLFF dataset",
    "mipnerf360": "MipNerf360 dataset",
    "d-nerf": "D-NeRF dataset",
    "visor": "VISOR dataset",
    "neu3d": "Neu3D dataset",
    "panoptic-sports": "Panoptic-Sports dataset",
    "nerfies": "Nerfies dataset",
    "iphone": "iPhone dataset",
    "monst3r": "MonST3R dataset",
    "iphone_som": "iPhone-SOM dataset",
}


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(
        sorted(method_descriptions.items(), key=lambda x: x[0])
    )
    return methods, method_descriptions


# all_datasets, all_descriptions = datasets_configs, datasets_descriptions
datasets_configs, datasets_descriptions = sort_methods(
    datasets_configs, datasets_descriptions
)

AnnotatedDatasetsConfigUnion = tyro.extras.subcommand_type_from_defaults(
    defaults=datasets_configs, descriptions=datasets_descriptions
)


#     #  TODO: scene specific config
#     if scene_name == "bicycle":
#         config["rotate_scene_x_axis_deg"] = -104
#         config["translate_scene_z"] = 0.1

#     if scene_name == "garden":
#         config["rotate_scene_x_axis_deg"] = -120
#         config["translate_scene_z"] = 0.2

#     if scene_name == "bonsai":
#         config["rotate_scene_x_axis_deg"] = -130
#         config["translate_scene_z"] = 0.25

#     if scene_name == "counter":
#         config["rotate_scene_x_axis_deg"] = -125
#         config["translate_scene_y"] = -0.1
#         config["translate_scene_z"] = 0.25

#     if scene_name == "kitchen":
#         config["rotate_scene_x_axis_deg"] = -130
#         config["translate_scene_z"] = 0.2

#     if scene_name == "room":
#         config["rotate_scene_x_axis_deg"] = -115

#     if scene_name == "stump":
#         config["rotate_scene_x_axis_deg"] = -137
#         config["translate_scene_z"] = 0.25
