import torch
import numpy as np
import torch.nn.functional as F
from typing import Union
from mvdatasets.geometry.common import (
    euclidean_to_homogeneous
)


def apply_rotation_3d(
    points_3d: Union[np.ndarray, torch.Tensor], rot: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies a 3D rotation to a set of points.

    Args:
        points_3d (numpy.ndarray or torch.Tensor): A (N, 3) array of 3D points.
        rot (numpy.ndarray or torch.Tensor): A (3, 3) rotation matrix or a batch (N, 3, 3) of rotation matrices.

    Returns:
        numpy.ndarray or torch.Tensor: A (N, 3) array of rotated 3D points.

    Raises:
        ValueError: If the shapes of `points_3d` or `rot` are invalid.
        TypeError: If the input types are inconsistent (mixing NumPy and PyTorch).
    """
    # Validate points_3d shape
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("`points_3d` must be a 2D array of shape (N, 3).")

    # Validate rotation matrix shape
    if rot.ndim == 2 and rot.shape == (3, 3):
        batched_rotation = False
    elif rot.ndim == 3 and rot.shape[1:] == (3, 3):
        batched_rotation = True
    else:
        raise ValueError("`rot` must be of shape (3, 3) or (N, 3, 3).")

    # Ensure consistent types between inputs
    if isinstance(points_3d, np.ndarray) and not isinstance(rot, np.ndarray):
        raise TypeError("Both inputs must be of the same type (NumPy or PyTorch).")
    if isinstance(points_3d, torch.Tensor) and not isinstance(rot, torch.Tensor):
        raise TypeError("Both inputs must be of the same type (NumPy or PyTorch).")

    # Apply rotation
    if isinstance(points_3d, np.ndarray):
        if batched_rotation:
            rotated_points = np.einsum("nij,nj->ni", rot, points_3d)
        else:
            rotated_points = points_3d @ rot.T
        return rotated_points
    elif isinstance(points_3d, torch.Tensor):
        if batched_rotation:
            rotated_points = torch.einsum("nij,nj->ni", rot, points_3d)
        else:
            rotated_points = points_3d @ rot.T
        return rotated_points
    

def apply_transformation_3d(
    points_3d: Union[np.ndarray, torch.Tensor],
    transform: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies a 3D affine transformation to a set of points.

    Args:
        points_3d (numpy.ndarray or torch.Tensor): A (N, 3) array of 3D points.
        transform (numpy.ndarray or torch.Tensor): A (4, 4) affine transformation matrix
                                                    or (N, 4, 4) for per-point transformations.

    Returns:
        numpy.ndarray or torch.Tensor: A (N, 3) array of transformed 3D points.

    Raises:
        ValueError: If the shapes of `points_3d` or `transform` are invalid.
        TypeError: If the input types are inconsistent (mixing NumPy and PyTorch).
    """
    # Check dimensionality of points_3d
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("`points_3d` must be a 2D array of shape (N, 3).")

    # Check dimensionality of transform
    if transform.ndim == 2 and transform.shape == (4, 4):
        batched_transform = False
    elif transform.ndim == 3 and transform.shape[1:] == (4, 4):
        batched_transform = True
    else:
        raise ValueError("`transform` must be of shape (4, 4) or (N, 4, 4).")

    # Ensure consistent types between inputs
    if isinstance(points_3d, np.ndarray) and not isinstance(transform, np.ndarray):
        raise TypeError("Both inputs must be of the same type (NumPy or PyTorch).")
    if isinstance(points_3d, torch.Tensor) and not isinstance(transform, torch.Tensor):
        raise TypeError("Both inputs must be of the same type (NumPy or PyTorch).")

    # Convert points_3d to homogeneous coordinates
    points_homogeneous = euclidean_to_homogeneous(points_3d)

    # Apply transformation
    if isinstance(points_3d, np.ndarray):
        if batched_transform:
            transformed_points = np.einsum("nij,nj->ni", transform, points_homogeneous)
        else:
            transformed_points = points_homogeneous @ transform.T
        return transformed_points[:, :3]
    elif isinstance(points_3d, torch.Tensor):
        if batched_transform:
            transformed_points = torch.einsum(
                "nij,nj->ni", transform, points_homogeneous
            )
        else:
            transformed_points = points_homogeneous @ transform.T
        return transformed_points[:, :3]


def pose_local_rotation(
    pose: Union[np.ndarray, torch.Tensor], rotation: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies a local rotation to the pose frame using a rotation matrix.

    Args:
        pose (numpy.ndarray or torch.Tensor): A 4x4 homogeneous transformation matrix representing the pose.
        rotation (numpy.ndarray or torch.Tensor): A 3x3 rotation matrix to be applied locally.

    Returns:
        numpy.ndarray or torch.Tensor: A 4x4 transformation matrix after applying the local rotation.

    Raises:
        ValueError: If `pose` is not a (4, 4) matrix or `rotation` is not a (3, 3) matrix.
        TypeError: If the input types are inconsistent (mixing NumPy and PyTorch).
    """
    # Check if inputs are from the same library
    if isinstance(pose, np.ndarray) and isinstance(rotation, np.ndarray):
        lib = np
    elif isinstance(pose, torch.Tensor) and isinstance(rotation, torch.Tensor):
        lib = torch
    else:
        raise TypeError(
            "Both `pose` and `rotation` must be either NumPy arrays or PyTorch tensors."
        )

    # Validate input shapes
    if pose.shape != (4, 4):
        raise ValueError("`pose` must be a 4x4 transformation matrix.")
    if rotation.shape != (3, 3):
        raise ValueError("`rotation` must be a 3x3 rotation matrix.")

    # Create a 4x4 rotation transform
    rotation_transform = lib.eye(4)  # Identity matrix of size 4x4
    rotation_transform[:3, :3] = rotation  # Embed 3x3 rotation in the top-left

    # Apply local rotation (pose is multiplied on the right)
    return pose @ rotation_transform


def pose_global_rotation(
    pose: Union[np.ndarray, torch.Tensor], rotation: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies a global rotation to the pose frame using a rotation matrix.

    Args:
        pose (numpy.ndarray or torch.Tensor): A 4x4 homogeneous transformation matrix representing the pose.
        rotation (numpy.ndarray or torch.Tensor): A 3x3 rotation matrix to be applied globally.

    Returns:
        numpy.ndarray or torch.Tensor: A 4x4 transformation matrix after applying the global rotation.

    Raises:
        ValueError: If `pose` is not a (4, 4) matrix or `rotation` is not a (3, 3) matrix.
        TypeError: If the input types are inconsistent (mixing NumPy and PyTorch).
    """
    # Check if inputs are from the same library
    if isinstance(pose, np.ndarray) and isinstance(rotation, np.ndarray):
        lib = np
    elif isinstance(pose, torch.Tensor) and isinstance(rotation, torch.Tensor):
        lib = torch
    else:
        raise TypeError(
            "Both `pose` and `rotation` must be either NumPy arrays or PyTorch tensors."
        )

    # Validate input shapes
    if pose.shape != (4, 4):
        raise ValueError("`pose` must be a 4x4 transformation matrix.")
    if rotation.shape != (3, 3):
        raise ValueError("`rotation` must be a 3x3 rotation matrix.")

    # Create a 4x4 rotation transform
    if lib == torch:
        rotation_transform = torch.eye(4, device=pose.device, dtype=pose.dtype)
    elif lib == np:
        rotation_transform = lib.eye(4)  # Identity matrix of size 4x4

    rotation_transform[:3, :3] = rotation  # Embed 3x3 rotation in the top-left

    # Apply global rotation (rotation is multiplied on the left)
    return rotation_transform @ pose