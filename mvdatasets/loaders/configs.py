import numpy as np
from pathlib import Path
from mvdatasets.utils.printing import print_error
from dataclasses import dataclass


@dataclass
class Config:
    # Default generic configuration

    datasets_path: Path = Path("/home/stefano/Data")  # Path to the datasets folder
    dataset_name: str = "dtu"  # Dataset name (e.g., "nerf_synthetic", "shelly", ...)
    scene_name: str = "dtu_scan83"  # Scene name (e.g. "lego", "khady", ...)
    auto_center: bool = False  # Shift the average of cameras centers to the origin
    rotate_deg: np.ndarray = np.array(
        [0.0, 0.0, 0.0]
    )  # Scene rotation angles in degrees
    max_cameras_distance: float = (
        None  # Maximum distance of the furthest camera from the origin (for scaling), if None, no scaling
    )
    foreground_scale_mult: float = (
        1.0  # Foreground area scale factor (<= 1.0), (1.0 = no scaling), e.g. 0.5, 1.0, 2.0, ...
    )
    subsample_factor: int = (
        1  # Subsampling factor (>= 1), (1 = no subsampling), e.g. 2, 3, 4, ...
    )
    init_sphere_radius_mult: float = (
        0.1  # Initial sphere radius multiplier (<= 1.0), (for SDF initialization)
    )
    pose_only: bool = False  # Load only poses (no images)

    def __post__init__(self):
        # Check configuration values

        # datasets_path
        if not self.datasets_path.exists():
            print_error(f"Dataset path {self.datasets_path} does not exist.")
        # dataset_name
        if type(self.dataset_name) is not str or len(self.dataset_name) == 0:
            print_error("dataset_name must be a non-empty string")
        # scene_name
        if type(self.scene_name) is not str or len(self.scene_name) == 0:
            print_error("scene_name must be a non-empty string")
        # auto_center
        if type(self.auto_center) is not bool:
            print_error("auto_center must be a boolean")
        # max_cameras_distance
        if self.max_cameras_distance is not None and self.max_cameras_distance <= 0:
            print_error("max_cameras_distance must be > 0 or None")
        # subsample factor
        if type(self.subsample_factor) is not int or self.subsample_factor < 1:
            print_error("subsample_factor must be an integer >= 1")
        # rotate_deg
        if type(self.rotate_deg) is not np.ndarray or len(self.rotate_deg) != 3:
            print_error("rotate_deg must be a numpy array of 3 elements")
        # multipliers
        if self.foreground_scale_mult <= 0 or self.foreground_scale_mult > 1:
            print_error("foreground_scale_mult must be in (0, 1]")
        if self.init_sphere_radius_mult <= 0 or self.init_sphere_radius_mult > 1:
            print_error("init_sphere_radius_mult must be in (0, 1]")
        # pose_only
        if type(self.pose_only) is not bool:
            print_error("pose_only must be a boolean")


@dataclass
class BlenderConfig(Config):
    # Default dataset configuration

    load_masks: bool = True  # Load mask images
    use_binary_mask: bool = True  # Convert masks to binary
    white_bg: bool = True  # Use white background (else black)
    test_skip: int = 20  # Skip every test_skip images from test split


def get_scene_preset(dataset_name: str, scene_name: str) -> dict:

    # test dtu
    if dataset_name == "dtu":
        # dataset specific config
        config = {
            "subsample_factor": 1,
        }

    # test blended-mvs
    elif dataset_name == "blended-mvs":
        # dataset specific config
        config = {}

    # test nerf_synthetic
    elif dataset_name == "nerf_synthetic":
        # dataset specific config
        config = {
            "test_skip": 20,
        }

    # test shelly
    elif dataset_name == "shelly":
        # dataset specific config
        config = {"test_skip": 4, "init_sphere_radius_mult": 0.2}

    # test nerf_furry
    elif dataset_name == "nerf_furry":
        # dataset specific config
        config = {
            "test_skip": 10,
        }

    # test dmsr
    elif dataset_name == "dmsr":
        # dataset specific config
        config = {
            "test_skip": 5,
        }

    # test refnerf
    elif dataset_name == "refnerf":
        # dataset specific config
        config = {
            "test_skip": 10,
        }

    # test ingp
    elif dataset_name == "ingp":
        # dataset specific config
        config = {}

    # test llff
    elif dataset_name == "llff":
        # dataset specific config
        config = {
            "scene_type": "unbounded",
        }

    # test mipnerf360
    elif dataset_name == "mipnerf360":

        # dataset specific config
        config = {
            "scene_type": "unbounded",
            "subsample_factor": 8,
        }

        # scene specific config
        if scene_name == "bicycle":
            config["rotate_scene_x_axis_deg"] = -104
            config["translate_scene_z"] = 0.1

        if scene_name == "garden":
            config["rotate_scene_x_axis_deg"] = -120
            config["translate_scene_z"] = 0.2

        if scene_name == "bonsai":
            config["rotate_scene_x_axis_deg"] = -130
            config["translate_scene_z"] = 0.25

        if scene_name == "counter":
            config["rotate_scene_x_axis_deg"] = -125
            config["translate_scene_y"] = -0.1
            config["translate_scene_z"] = 0.25

        if scene_name == "kitchen":
            config["rotate_scene_x_axis_deg"] = -130
            config["translate_scene_z"] = 0.2

        if scene_name == "room":
            config["rotate_scene_x_axis_deg"] = -115

        if scene_name == "stump":
            config["rotate_scene_x_axis_deg"] = -137
            config["translate_scene_z"] = 0.25

    # test d-nerf
    elif dataset_name == "d-nerf":
        # dataset specific config
        config = {}

    # test visor
    elif dataset_name == "visor":
        # dataset specific config
        config = {}

    # test neu3d
    elif dataset_name == "neu3d":
        # dataset specific config
        config = {}

    # test panoptic-sports
    elif dataset_name == "panoptic-sports":
        # dataset specific config
        config = {}

    # test iphone
    elif dataset_name == "iphone":
        # dataset specific config
        config = {}

    # test monst3r
    elif dataset_name == "monst3r":
        # dataset specific config
        config = {}

    else:
        # undefined empty config
        config = {}

    return config
