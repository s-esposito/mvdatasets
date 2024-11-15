import numpy as np
from copy import deepcopy
import torch

# def contract_points(points):
#     """
#     Warping function that smoothly maps
#     all coordinates outside of a ball of
#     radius 0.5 into a ball of radius 1.
#     From MipNeRF360.
#     """

#     points_ = deepcopy(points)

#     # compute points norm
#     if isinstance(points_, np.ndarray):
#         points_norm = np.linalg.norm(points_, axis=1)
#     elif isinstance(points_, torch.Tensor):
#         points_norm = torch.norm(points_, dim=1)
#     else:
#         raise ValueError("points must be a numpy array or a torch tensor")

#     # find points to contract
#     to_contract = (points_norm > 0.5)

#     # if norm <= 0.5, do nothing
#     # if norm > 0.5, contract the point to the unit sphere

#     contracted_points = (1.0 - 0.25/points_norm[to_contract])[:, None] * (points[to_contract] / points_norm[to_contract][:, None])
#     points_[to_contract] = contracted_points
#     return points_


def contract_points(points):
    """
    Warping function that smoothly maps
    all coordinates outside of a ball of
    radius 0.5 into a ball of radius 1.
    From MipNeRF360.
    """

    # TODO: contract dt and z too given a ray_samples_packed object

    scale = 2

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


def uncontract_points(points):
    """
    Inverse of contract_points.
    """

    # TODO: uncontract dt and z too given a ray_samples_packed object
    scale = 2
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
