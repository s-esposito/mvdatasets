from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union
from mvdatasets.configs.base_config import PrintableConfig
from mvdatasets.configs.machine_config import MachineConfig
from mvdatasets.configs.datasets_configs import AnnotatedDatasetsConfigUnion
from mvdatasets.configs.datasets_configs import datasets_configs
from mvdatasets.utils.printing import print_warning


@dataclass
class ExampleConfig(PrintableConfig):
    """Example configuration."""

    # dataset configuration
    data: AnnotatedDatasetsConfigUnion
    """Dataset configuration"""

    scene_name: Optional[str] = None
    """Name of the scene (e.g., "dtu_scan83", "khady", ...)"""

    # paths
    datasets_path: Path = Path("data")
    """Relative or absolute path to the root datasets directory"""
    output_path: Path = Path("outputs")
    """Relative or absolute path to the output directory to save splots, videos, etc..."""

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
        # if scene_name is None, print warning 
        if self.scene_name is None:
            print_warning("scene_name is None, using preset test scene for dataset")
