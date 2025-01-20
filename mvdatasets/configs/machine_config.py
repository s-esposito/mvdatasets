from pathlib import Path
from typing import Tuple, Any, Type, Literal, List, Optional  # Machine related configs
from dataclasses import dataclass, field
from mvdatasets.configs.base_config import PrintableConfig


@dataclass
class MachineConfig(PrintableConfig):
    """Configuration of machine setup"""

    seed: int = 42
    """random seed initialization"""

    device: Literal["cpu", "cuda", "mps"] = "cuda"
    """device type to use for training"""

    def __post__init__(self):

        import torch
        import numpy as np
        import random

        # Set a random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Check if CUDA (GPU support) is available
        if "cuda" in self.device:
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available, change device to 'cpu'")
            else:
                # Set a random seed for GPU
                torch.cuda.manual_seed(self.seed)
