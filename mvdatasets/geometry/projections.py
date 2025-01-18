import torch
import numpy as np
import torch.nn.functional as F
from typing import Union
from mvdatasets.geometry.common import (
    euclidean_to_homogeneous,
    homogeneous_to_euclidean,
)
from mvdatasets.geometry.rigid import apply_transformation_3d


def local_perspective_projection(
    intrinsics: Union[np.ndarray, torch.Tensor],
    points_3d_camera: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply perspective projection to 3D points.

    Args:
        intrinsics (np.ndarray or torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        points_3d_camera (np.ndarray or torch.Tensor): Array of 3D points of shape (N, 3).

    Returns:
        np.ndarray or torch.Tensor: Projected 2D points of shape (N, 2).

    Raises:
        ValueError: If inputs have invalid shapes or types.
    """
    if points_3d_camera.shape[-1] != 3:
        raise ValueError("`points_3d_camera` must have shape (N, 3).")
    if intrinsics.shape != (3, 3):
        raise ValueError("`intrinsics` must have shape (3, 3).")

    augmented_points_3d_camera = euclidean_to_homogeneous(points_3d_camera)

    if isinstance(intrinsics, torch.Tensor):
        K0 = torch.cat(
            [
                intrinsics,
                torch.zeros((3, 1), device=intrinsics.device, dtype=intrinsics.dtype),
            ],
            dim=1,
        )
    elif isinstance(intrinsics, np.ndarray):
        K0 = np.concatenate(
            [intrinsics, np.zeros((3, 1), dtype=intrinsics.dtype)], axis=1
        )
    else:
        raise TypeError("`intrinsics` must be either a numpy.ndarray or torch.Tensor.")

    homogeneous_points_2d_screen = (K0 @ augmented_points_3d_camera.T).T
    points_2d_screen = homogeneous_to_euclidean(homogeneous_points_2d_screen)

    return points_2d_screen


def local_inv_perspective_projection(
    intrinsics_inv: Union[np.ndarray, torch.Tensor],
    points_2d_screen: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply inverse perspective projection to 2D screen points.

    Args:
        intrinsics_inv (np.ndarray or torch.Tensor): Inverse of camera intrinsic matrix of shape (N, 3, 3) or (3, 3).
        points_2d_screen (np.ndarray or torch.Tensor): 2D points in screen coordinates of shape (N, 2).

    Returns:
        np.ndarray or torch.Tensor: Unprojected 3D points of shape (N, 3).

    Raises:
        ValueError: If inputs have invalid shapes or types.
    """

    # check input shapes
    if intrinsics_inv.ndim == 2:
        intrinsics_inv = intrinsics_inv[None, ...]  # Add batch dimension
    elif intrinsics_inv.ndim == 3:
        pass
    else:
        raise ValueError(
            f"intrinsics_inv: {intrinsics_inv.shape} must have shape (N, 3, 3) or (3, 3)."
        )

    if intrinsics_inv.shape[1:] != (3, 3):
        raise ValueError(
            f"intrinsics_inv: {intrinsics_inv.shape} must have shape (N, 3, 3) or (3, 3)."
        )

    if (
        intrinsics_inv.shape[0] != points_2d_screen.shape[0]
        and intrinsics_inv.shape[0] != 1
    ):
        raise ValueError(
            f"input shapes do not match: intrinsics_inv: {intrinsics_inv.shape} and points_2d_screen: {points_2d_screen.shape}."
        )

    if points_2d_screen.ndim == 2 and points_2d_screen.shape[-1] != 2:
        raise ValueError("`points_2d_screen` must have shape (N, 2).")

    augmented_points_2d_screen = euclidean_to_homogeneous(points_2d_screen)  # (N, 3)
    augmented_points_2d_screen = augmented_points_2d_screen[..., None]  # (N, 3, 1)
    augmented_points_3d_camera = (
        intrinsics_inv @ augmented_points_2d_screen
    )  # (N, 3, 3) @ (N, 3, 1)
    # reshape to (N, 3)
    augmented_points_3d_camera = augmented_points_3d_camera.squeeze(-1)  # (N, 3)

    return augmented_points_3d_camera


def global_perspective_projection(
    intrinsics: Union[np.ndarray, torch.Tensor],
    c2w: Union[np.ndarray, torch.Tensor],
    points_3d_world: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Projects 3D points to 2D screen space using camera intrinsics and pose.

    Args:
        intrinsics (np.ndarray or torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        c2w (np.ndarray or torch.Tensor): Camera-to-world transformation matrix of shape (4, 4).
        points_3d_world (np.ndarray or torch.Tensor): 3D points in world space of shape (N, 3).

    Returns:
        points_2d_screen (np.ndarray or torch.Tensor): 2D screen points of shape (N, 2).
        in_front_of_camera_mask (np.ndarray or torch.Tensor): Boolean mask indicating points in front of the camera.

    Raises:
        ValueError: If inputs have incorrect types or shapes.
    """
    if isinstance(points_3d_world, torch.Tensor):
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.tensor(
                intrinsics, dtype=torch.float32, device=points_3d_world.device
            )
        if not isinstance(c2w, torch.Tensor):
            c2w = torch.tensor(c2w, dtype=torch.float32, device=points_3d_world.device)
        w2c = torch.inverse(c2w)  # World-to-camera transformation
    elif isinstance(points_3d_world, np.ndarray):
        if not isinstance(intrinsics, np.ndarray):
            intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if not isinstance(c2w, np.ndarray):
            c2w = np.asarray(c2w, dtype=np.float32)
        w2c = np.linalg.inv(c2w)
    else:
        raise ValueError(
            "`points_3d_world` must be either a torch.Tensor or np.ndarray."
        )

    # Transform 3D points from world space to camera space
    points_3d_camera = apply_transformation_3d(points_3d_world, w2c)

    # Get points in front of the camera (z > 0)
    in_front_of_camera_mask = points_3d_camera[..., 2] > 0

    # Project 3D camera space points to 2D screen space
    points_2d_screen = local_perspective_projection(intrinsics, points_3d_camera)

    return points_2d_screen, in_front_of_camera_mask


def global_inv_perspective_projection(
    intrinsics_inv: Union[np.ndarray, torch.Tensor],
    c2w: Union[np.ndarray, torch.Tensor],
    points_2d_screen: Union[np.ndarray, torch.Tensor],
    depth: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Unprojects 2D screen points to 3D world space using camera intrinsics and pose.

    Args:
        intrinsics_inv (np.ndarray or torch.Tensor): Inverse of the camera intrinsic matrix of shape (3, 3).
        c2w (np.ndarray or torch.Tensor): Camera-to-world transformation matrix of shape (4, 4).
        points_2d_screen (np.ndarray or torch.Tensor): 2D screen points of shape (N, 2).
        depth (np.ndarray or torch.Tensor): Depth values for the points, shape (N,).

    Returns:
        np.ndarray or torch.Tensor: Unprojected 3D world points of shape (N, 3).

    Raises:
        ValueError: If inputs have incompatible types or shapes.
    """

    # Validate input shapes
    if points_2d_screen.shape[0] != depth.shape[0]:
        raise ValueError(
            f"input shapes do not match: points_2d_screen: {points_2d_screen.shape} and depth: {depth.shape}."
        )
    if points_2d_screen.shape[1] != 2:
        raise ValueError(
            f"points_2d_screen: {points_2d_screen} must have shape (N, 2)."
        )
    if depth.ndim != 1:
        raise ValueError(f"depth: {depth.shape} must be a 1D array.")

    # Convert intrinsics and c2w to the same type as points_2d_screen
    if isinstance(points_2d_screen, np.ndarray):
        intrinsics_inv = np.asarray(intrinsics_inv, dtype=np.float32)
        c2w = np.asarray(c2w, dtype=np.float32)
    elif isinstance(points_2d_screen, torch.Tensor):
        if not isinstance(intrinsics_inv, torch.Tensor):
            intrinsics_inv = torch.tensor(
                intrinsics_inv, dtype=torch.float32, device=points_2d_screen.device
            )
        if not isinstance(c2w, torch.Tensor):
            c2w = torch.tensor(c2w, dtype=torch.float32, device=points_2d_screen.device)
    else:
        raise ValueError("`points_2d_screen` must be a torch.Tensor or np.ndarray.")

    # Ray origin is the camera center
    rays_o = c2w[:3, -1]  # Extract camera center from the last column of c2w
    if isinstance(rays_o, torch.Tensor):
        rays_o = rays_o[None, ...]  # Add batch dimension for consistency
    else:
        rays_o = np.expand_dims(rays_o, axis=0)

    # Unproject 2D screen points to camera space
    points_3d_camera = local_inv_perspective_projection(
        intrinsics_inv,
        points_2d_screen,
    )

    # multiply by depth
    points_3d_camera *= depth[..., None]

    # Transform points from camera space to world space
    points_3d_world = (c2w[:3, :3] @ points_3d_camera.T).T

    # # Normalize the direction vectors
    # if isinstance(points_3d_world, torch.Tensor):
    #     # rays_d = F.normalize(points_3d_world, dim=-1)
    #     rays_d = points_3d_world / torch.norm(points_3d_world, dim=-1, keepdim=True)
    # else:
    #     rays_d = points_3d_world / np.linalg.norm(
    #         points_3d_world, axis=-1, keepdims=True
    #     )

    # # Scale direction vectors by depth
    # points_3d_world = rays_d * depth[..., None]

    # Add ray origin to scale and translate points
    points_3d_world += rays_o

    return points_3d_world
