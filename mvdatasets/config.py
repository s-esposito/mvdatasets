import numpy as np
from pathlib import Path
from mvdatasets.utils.printing import print_error
from dataclasses import dataclass, field
from typing import Optional, List

# 
valid_scene_types = ["bounded", "unbounded"]

@dataclass
class DatasetConfig:
    # Default generic configuration

    splits: List[str] = field(default_factory=lambda: ["train"])  # Dataset splits
    auto_center: bool = False  # Shift the average of cameras centers to the origin
    rotate_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Scene rotation angles in degrees
    max_cameras_distance: Optional[float] = None  # Maximum distance of the furthest camera from the origin (for scaling), if None, no scaling
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
        #
        self.rotate_deg = np.array(self.rotate_deg, dtype=np.float32)
        
        # Check configuration values

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
            

# Static Datasets Configurations ------------------------------------------------


@dataclass
class BlenderConfig(DatasetConfig):
    # Default dataset configuration

    scene_type: str = "bounded"  # Type of scene (bounded or unbounded)
    load_masks: bool = True  # Load mask images
    use_binary_mask: bool = True  # Convert masks to binary
    white_bg: bool = True  # Use white background (else black)
    test_skip: int = 20  # Skip every test_skip images from test split
    
    def __post__init__(self):
        #
        super().__post__init__()
        #
        if self.scene_type not in valid_scene_types:
            print_error(f"scene_type must be one of {valid_scene_types}")
        # load_masks
        if type(self.load_masks) is not bool:
            print_error("load_masks must be a boolean")
        # use_binary_mask
        if type(self.use_binary_mask) is not bool:
            print_error("use_binary_mask must be a boolean")
        # white_bg
        if type(self.white_bg) is not bool:
            print_error("white_bg must be a boolean")
        # test_skip
        if type(self.test_skip) is not int or self.test_skip < 1:
            print_error("test_skip must be an integer >= 1")
        # check if splits are valid
        for split in self.splits:
            if split not in ["train", "test"]:
                print_error(f"split {split} not supported")

    
# Dynamic Datasets Configurations -----------------------------------------------