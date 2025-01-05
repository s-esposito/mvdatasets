import torch
import numpy as np
import torch.nn.functional as F
from typing import Union


def convert_6d_to_rotation_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def get_min_max_cameras_distances(poses: list) -> tuple:
    """
    return maximum pose distance from origin

    Args:
        poses (list): list of numpy (4, 4) poses

    Returns:
        min_dist (float): minumum camera distance from origin
        max_dist (float): maximum camera distance from origin
    """
    if len(poses) == 0:
        raise ValueError("poses list empty")
    
    # get all camera centers
    camera_centers = np.stack(poses, 0)[:, :3, 3]
    camera_distances_from_origin = np.linalg.norm(camera_centers, axis=1)

    min_dist = np.min(camera_distances_from_origin)
    max_dist = np.max(camera_distances_from_origin)

    return min_dist, max_dist


def rotation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (np.random.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (
        np.eye(3)
        + skew_sym_mat
        + np.dot(skew_sym_mat, skew_sym_mat) * ((1 - c) / (s**2 + 1e-8))
    )


def deg2rad(
    deg: Union[float, np.ndarray, torch.Tensor]
) -> Union[float, np.ndarray, torch.Tensor]:
    return deg * np.pi / 180


def scale_3d(scale: float) -> np.ndarray:
    return np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])


def rot_x_3d(theta: float, device: str = None) -> Union[np.ndarray, torch.Tensor]:
    if device is None:
        # numpy
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float32,
        )
    else:
        # torch
        return torch.tensor(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ],
            device=device,
            dtype=torch.float32,
        )


def rot_y_3d(theta: float, device: str = None) -> Union[np.ndarray, torch.Tensor]:
    if device is None:
        # numpy
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ],
            dtype=np.float32,
        )
    else:
        # torch
        return torch.tensor(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ],
            device=device,
            dtype=torch.float32,
        )


def rot_z_3d(theta: float, device: str = None) -> Union[np.ndarray, torch.Tensor]:
    if device is None:
        # numpy
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
    else:
        # torch
        return torch.tensor(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )


def rot_euler_3d(
    theta_x: float, theta_y: float, theta_z: float, device: str = None
) -> Union[np.ndarray, torch.Tensor]:
    """angles are in radians"""
    return (
        rot_z_3d(theta_z, device)
        @ rot_y_3d(theta_y, device)
        @ rot_x_3d(theta_x, device)
    )


def rot_euler_3d_deg(
    theta_x: float, theta_y: float, theta_z: float, device: str = None
) -> Union[np.ndarray, torch.Tensor]:
    """angles are in degrees"""
    return rot_euler_3d(deg2rad(theta_x), deg2rad(theta_y), deg2rad(theta_z), device)


def opengl_matrix_world_from_w2c(w2c: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV camera-to-world pose to OpenGL camera pose.

    Args:
    - w2c: A 4x4 np.ndarray representing the OpenCV world-to-camera pose.

    Returns:
    - A 4x4 np.ndarray representing the OpenGL camera pose.
    """
    # flip_z = np.diag(np.array([1, 1, -1, 1]))
    # return np.matmul(c2w, flip_z)

    # Flip Y-axis
    w2c[1, :] = -w2c[1, :]  # Invert the z-axis

    # Convert from OpenCV coordinate system (right-handed, y-up, z-forward)
    # to OpenGL/three.js coordinate system (right-handed, y-up, z-backward)
    w2c[2, :] = -w2c[2, :]  # Invert the z-axis

    c2w = np.linalg.inv(w2c)

    return c2w


def opengl_projection_matrix_from_intrinsics(
    K: np.ndarray, width: int, height: int, near: int, far: int
) -> np.ndarray:
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    projection_matrix = np.array(
        [
            [2.0 * fx / width, 0.0, -(2.0 * cx / width - 1.0), 0.0],
            [0.0, 2.0 * fy / height, -(2.0 * cy / height - 1.0), 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    # Transpose to column-major order for OpenGL
    return projection_matrix


# def pad_matrix(
#     matrix: Union[np.ndarray, torch.Tensor]
# ) -> Union[np.ndarray, torch.Tensor]:
#     """
#     Pads a transformation matrix with a homogeneous bottom row [0, 0, 0, 1].

#     Args:
#         matrix (np.ndarray or torch.Tensor): A (3, 4) or (N, 3, 4) transformation matrix.

#     Returns:
#         np.ndarray or torch.Tensor: A (4, 4) or (N, 4, 4) transformation matrix with the bottom row added.

#     Raises:
#         ValueError: If `matrix` is not a valid 2D or 3D transformation matrix.
#     """
#     if isinstance(matrix, np.ndarray):
#         if matrix.ndim == 2 and matrix.shape == (3, 4):  # Single matrix case
#             bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype)
#             padded_matrix = np.vstack([matrix, bottom[None, :]])
#         elif matrix.ndim == 3 and matrix.shape[1:] == (3, 4):  # Batch case
#             bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype)
#             bottom = np.tile(bottom, (matrix.shape[0], 1, 1))  # Expand for batch
#             padded_matrix = np.concatenate([matrix, bottom[:, None, :]], axis=1)
#         else:
#             raise ValueError("Invalid matrix shape. Expected (3, 4) or (N, 3, 4).")
#     elif isinstance(matrix, torch.Tensor):
#         if matrix.ndim == 2 and matrix.shape == (3, 4):  # Single matrix case
#             bottom = torch.tensor(
#                 [0.0, 0.0, 0.0, 1.0], device=matrix.device, dtype=matrix.dtype
#             )
#             padded_matrix = torch.cat([matrix, bottom[None, :]], dim=0)
#         elif matrix.ndim == 3 and matrix.shape[1:] == (3, 4):  # Batch case
#             bottom = torch.tensor(
#                 [0.0, 0.0, 0.0, 1.0], device=matrix.device, dtype=matrix.dtype
#             )
#             bottom = bottom.expand(matrix.shape[0], 1, 4)  # Expand for batch
#             padded_matrix = torch.cat([matrix, bottom], dim=1)
#         else:
#             raise ValueError("Invalid matrix shape. Expected (3, 4) or (N, 3, 4).")
#     else:
#         raise ValueError(
#             "Unsupported matrix type, should be np.ndarray or torch.Tensor."
#         )

#     return padded_matrix


# def unpad_matrix(
#     matrix: Union[np.ndarray, torch.Tensor]
# ) -> Union[np.ndarray, torch.Tensor]:
#     """
#     Removes the homogeneous bottom row from a padded transformation matrix.

#     Args:
#         matrix (np.ndarray or torch.Tensor): A (4, 4) or (N, 4, 4) transformation matrix.

#     Returns:
#         np.ndarray or torch.Tensor: A (3, 4) or (N, 3, 4) transformation matrix.

#     Raises:
#         ValueError: If `matrix` does not have the correct shape.
#     """
#     if matrix.ndim == 2:
#         if matrix.shape != (4, 4):
#             raise ValueError("Invalid matrix shape. Expected (4, 4).")
#         return matrix[:3, :]  # Single matrix case
#     elif matrix.ndim == 3:
#         if matrix.shape[1:] != (4, 4):
#             raise ValueError("Invalid batch matrix shape. Expected (N, 4, 4).")
#         return matrix[:, :3, :]  # Batch case
#     else:
#         raise ValueError("Unsupported matrix dimensionality, should be 2D or 3D.")


def euclidean_to_homogeneous(
    points: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts Euclidean coordinates to homogeneous coordinates by appending a column of ones.

    Args:
        points (np.ndarray or torch.Tensor): A 2D array of shape (N, C) representing Euclidean points.

    Returns:
        np.ndarray or torch.Tensor: A 2D array of shape (N, C+1) in homogeneous coordinates.

    Raises:
        TypeError: If `points` is not a NumPy array or PyTorch tensor.
        ValueError: If `points` is not a 2D array.
    """
    # Check if input is a 2D array
    if points.ndim != 2:
        raise ValueError("`points` must be a 2D array of shape (N, C).")

    if isinstance(points, np.ndarray):
        ones = np.ones((points.shape[0], 1))
        return np.hstack((points, ones))
    elif isinstance(points, torch.Tensor):
        ones = torch.ones(
            (points.shape[0], 1), dtype=points.dtype, device=points.device
        )
        return torch.cat((points, ones), dim=1)
    else:
        raise TypeError("`points` must be either a numpy.ndarray or torch.Tensor.")


def homogeneous_to_euclidean(
    vectors: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts homogeneous coordinates to Euclidean coordinates by dividing by the last coordinate.

    Args:
        vectors (np.ndarray or torch.Tensor): An (N, C+1) array of homogeneous coordinates.

    Returns:
        np.ndarray or torch.Tensor: An (N, C) array of Euclidean coordinates.

    Raises:
        ValueError: If the input does not have at least two dimensions or if `vectors` is not
                    a NumPy array or PyTorch tensor.
    """
    # Ensure input has at least 2 dimensions
    if vectors.shape[-1] < 2:
        raise ValueError("Input `vectors` must have at least two dimensions (N, C+1).")

    # Perform division by the last coordinate
    return vectors[..., :-1] / vectors[..., -1:]


def look_at(eye, center, up):
    """
    Compute camera pose from look-at vectors (OpenCV format).

    Args:
        eye (np.ndarray): (3,) Camera position in world space.
        center (np.ndarray): (3,) Point to look at in world space.
        up (np.ndarray): (3,) Up vector.

    Returns:
        np.ndarray: (4, 4) Camera pose matrix.
    """
    assert eye.shape == (3,)
    assert center.shape == (3,)
    assert up.shape == (3,)

    # Compute forward vector
    forward = center - eye  # vector from eye to center
    forward = forward / np.linalg.norm(forward)

    # Compute right and up vectors
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    new_up = np.cross(forward, right)
    new_up = new_up / np.linalg.norm(new_up)

    # Construct rotation matrix
    rotation = np.eye(4)
    rotation[:3, 0] = -right
    rotation[:3, 1] = -new_up
    rotation[:3, 2] = forward

    # Add translation
    rotation[:3, 3] = eye

    return rotation


# def look_at(
#     camera_origin: np.ndarray, target_point: np.ndarray, up: np.ndarray
# ) -> np.ndarray:
#     """Compute camera pose from look at vectors
#     args:
#         camera_origin (np.ndarray) : (3,) camera position
#         target_point (np.ndarray) : (3,) point to look at
#         up (np.ndarray) : (3,) up vector in world space
#     out:
#         pose (np.ndarray) : (4, 4) camera pose
#     """

#     assert camera_origin.shape == (3,)
#     assert target_point.shape == (3,)
#     assert up.shape == (3,)

#     # get camera frame
#     z = camera_origin - target_point
#     z = z / np.linalg.norm(z)
#     x = np.cross(up, z)
#     if np.linalg.norm(x) == 0:
#         raise ValueError("up vector is parallel to camera direction")
#     x = x / np.linalg.norm(x)
#     y = np.cross(z, x)
#     if np.linalg.norm(y) == 0:
#         raise ValueError("up vector is parallel to camera direction")
#     y = y / np.linalg.norm(y)

#     # get rotation matrix
#     rotation = np.eye(3)
#     rotation[:, 0] = x
#     rotation[:, 1] = y
#     rotation[:, 2] = z

#     # add translation
#     pose = np.eye(4)
#     pose[:3, :3] = rotation
#     pose[:3, 3] = camera_origin

#     return pose


def get_mask_points_in_image_range(
    points_2d_screen: Union[np.ndarray, torch.Tensor], width: int, height: int
) -> Union[np.ndarray, torch.Tensor]:
    """Filter out points that are outside the image."""
    mask = (points_2d_screen[:, 0] >= 0) & (points_2d_screen[:, 0] < width)
    mask &= (points_2d_screen[:, 1] >= 0) & (points_2d_screen[:, 1] < height)
    return mask


####################################################
# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/utils/colmap_parsing_utils.py#L454


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


####################################################
