from pathlib import Path
from typing import Tuple, Any, Type, Literal, List, Optional
import numpy as np
import tyro
from dataclasses import dataclass, field
from mvdatasets.configs.base_config import PrintableConfig

# TODO: use enums for scene types
# from enum import Enum

#
SCENE_TYPES = ["bounded", "unbounded"]


@dataclass
class DatasetConfig(PrintableConfig):
    """General dataset configuration"""

    # Default generic configuration

    dataset_name: tyro.conf.Suppress[str] = ""
    """Name of the dataset (e.g., "dtu", "shelly", ...)"""
    scene_type: str = "unbounded"
    """Type of scene (bounded or unbounded)"""
    splits: List[str] = field(default_factory=lambda: ["train"])
    """Dataset splits"""
    auto_center: bool = False
    """Shift the average of cameras centers to the origin"""
    rotate_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Scene rotation angles in degrees"""
    max_cameras_distance: Optional[float] = None
    """Maximum distance of the furthest camera from the origin (for scaling), if None, no scaling"""
    foreground_scale_mult: float = 1.0
    """Foreground area scale factor (<= 1.0), (1.0 = no scaling), e.g. 0.5, 1.0, 2.0, ..."""
    subsample_factor: int = 1
    """Subsampling factor (>= 1), (1 = no subsampling), e.g. 2, 3, 4, ..."""
    init_sphere_radius_mult: float = 0.1
    """Initial sphere radius multiplier (<= 1.0), (for SDF initialization)"""
    pose_only: bool = False
    """Load only poses (no images)"""

    def __post__init__(self):
        #
        self.rotate_deg = np.array(self.rotate_deg, dtype=np.float32)

        # Check configuration values
        # dataset_name
        if type(self.dataset_name) is not str or len(self.dataset_name) == 0:
            raise ValueError("dataset_name must be a non-empty string")
        # scene_type
        if self.scene_type not in SCENE_TYPES:
            raise ValueError(f"scene_type must be one of {SCENE_TYPES}")
        # auto_center
        if type(self.auto_center) is not bool:
            raise ValueError("auto_center must be a boolean")
        # max_cameras_distance
        if self.max_cameras_distance is not None and self.max_cameras_distance <= 0:
            raise ValueError("max_cameras_distance must be > 0 or None")
        # subsample factor
        if type(self.subsample_factor) is not int or self.subsample_factor < 1:
            raise ValueError("subsample_factor must be an integer >= 1")
        # rotate_deg
        if type(self.rotate_deg) is not np.ndarray or len(self.rotate_deg) != 3:
            raise ValueError("rotate_deg must be a numpy array of 3 elements")
        # multipliers
        if self.foreground_scale_mult <= 0 or self.foreground_scale_mult > 1:
            raise ValueError("foreground_scale_mult must be in (0, 1]")
        if self.init_sphere_radius_mult <= 0 or self.init_sphere_radius_mult > 1:
            raise ValueError("init_sphere_radius_mult must be in (0, 1]")
        # pose_only
        if type(self.pose_only) is not bool:
            raise ValueError("pose_only must be a boolean")
