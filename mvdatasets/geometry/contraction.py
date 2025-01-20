import numpy as np
import torch
from typing import Union


def contract_points(points, scale=2) -> Union[np.ndarray, torch.Tensor]:
    """
    Warping function that smoothly maps all coordinates outside of a ball of radius 0.5 into a ball of radius 1.
    From :cite:t:`barron2022mipnerf360`.

    Args:
        points: (N, 3) numpy array or torch tensor
        scale: float
    Returns:
        (N, 3) numpy array or torch tensor of contracted points
    """
    # compute points norm
    if isinstance(points, np.ndarray):
        points_norm = np.linalg.norm(points * scale, axis=1)[:, np.newaxis]  # (N, 1)
        return np.where(
            points_norm < 1.0, points, (2 - 1.0 / points_norm) * points / points_norm
        )
    elif isinstance(points, torch.Tensor):
        points_norm = torch.norm(points * scale, dim=1, keepdim=True)
        return torch.where(
            points_norm < 1.0, points, (2 - 1.0 / points_norm) * points / points_norm
        )
    else:
        raise ValueError("points must be a numpy array or a torch tensor")


def uncontract_points(points, scale=2) -> Union[np.ndarray, torch.Tensor]:
    """
    Inverse of contract_points.

    Args:
        points: (N, 3) numpy array or torch tensor
        scale: float
    Returns:
        (N, 3) numpy array or torch tensor of uncontracted points
    """
    # compute points norm
    if isinstance(points, np.ndarray):
        points_norm = np.linalg.norm(points * scale, axis=1)[:, np.newaxis]  # (N, 1)
        return np.where(
            points_norm < 1.0, points, 1.0 / (2 - points_norm) * (points / points_norm)
        )
    elif isinstance(points, torch.Tensor):
        points_norm = torch.norm(points * scale, dim=1, keepdim=True)
        return torch.where(
            points_norm < 1.0, points, 1.0 / (2 - points_norm) * (points / points_norm)
        )
    else:
        raise ValueError("points must be a numpy array or a torch tensor")
