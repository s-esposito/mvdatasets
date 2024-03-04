import numpy as np
import torch


def contract_points(points):
    """
    Warping function that smoothly maps
    all coordinates outside of a ball of
    radius 0.5 into a ball of radius 1.
    From MipNeRF360.
    """
    
    # compute points norm
    if isinstance(points, np.ndarray):
        points_norm = np.linalg.norm(points, axis=1)
    elif isinstance(points, torch.Tensor):
        points_norm = torch.norm(points, dim=1)
    else:
        raise ValueError("points must be a numpy array or a torch tensor")
    
    # find points to contract
    to_contract = (points_norm > 0.5)
    contracted_points = (1 - 0.5/points_norm[to_contract])[:, None] * (points[to_contract] / points_norm[to_contract][:, None])
    points[to_contract] = contracted_points
    return points