from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union
from mvdatasets.configs.base_config import PrintableConfig
from mvdatasets.configs.machine_config import MachineConfig
from mvdatasets.configs.datasets_configs import AnnotatedDatasetsConfigUnion
from mvdatasets.configs.datasets_configs import datasets_configs


@dataclass
class ExampleConfig(PrintableConfig):
    """Example configuration."""

    scene_name: Optional[str] = None
    """Name of the scene (e.g., "dtu_scan83", "khady", ...)"""

    # paths
    datasets_path: Path = Path("/home/stefano/Data")
    """Relative or absolute path to the root datasets directory"""
    output_path: Path = Path("plots")
    """Relative or absolute path to the output directory to save splots, videos, etc..."""

    # dataset configuration
    data: AnnotatedDatasetsConfigUnion = field(
        default_factory=lambda: datasets_configs["nerf_synthetic"]
    )
    """Dataset configuration"""

    with_viewer: bool = False
    """Show viewers to visualize the examples"""

    # nested configs
    machine: MachineConfig = field(default_factory=MachineConfig)
    """Machine configuration"""

    def __post_init__(self):
        # datasets_path
        if not self.datasets_path.exists():
            raise ValueError(f"Dataset path {self.datasets_path} does not exist.")
        # create output directory if it does not exist
        self.output_path.mkdir(parents=True, exist_ok=True)
