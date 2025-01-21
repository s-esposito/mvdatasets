from rich import print
from pathlib import Path
from typing import Tuple, Any, Type, Literal, List, Optional
import numpy as np
from dataclasses import dataclass, field, asdict


@dataclass
class PrintableConfig:
    """Printable Config defining str function"""

    def asdict(self):
        return asdict(self)

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)