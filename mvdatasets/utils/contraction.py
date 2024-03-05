import numpy as np
from copy import deepcopy
import torch


def contract_points(points):
    """
    Warping function that smoothly maps
    all coordinates outside of a ball of
    radius 0.5 into a ball of radius 1.
    From MipNeRF360.
    """
    
    points_ = deepcopy(points)
    
    # compute points norm
    if isinstance(points_, np.ndarray):
        points_norm = np.linalg.norm(points_, axis=1)
    elif isinstance(points_, torch.Tensor):
        points_norm = torch.norm(points_, dim=1)
    else:
        raise ValueError("points must be a numpy array or a torch tensor")
    
    # find points to contract
    to_contract = (points_norm > 0.5)
    
    # if norm <= 0.5, do nothing
    # if norm > 0.5, contract the point to the unit sphere
    
    contracted_points = (1.0 - 0.25/points_norm[to_contract])[:, None] * (points[to_contract] / points_norm[to_contract][:, None])
    points_[to_contract] = contracted_points
    return points_