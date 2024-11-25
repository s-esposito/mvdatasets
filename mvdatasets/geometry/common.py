import torch
import numpy as np
import torch.nn.functional as F
from typing import Union


def convert_6d_to_rotation_matrix(cont_6d: torch.Tensor) -> torch.Tensor:
    """
    :param 6d vector (*, 6)
    :returns matrix (*, 3, 3)
    """
    # Extract the two 3D components from the 6D input
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]

    # Normalize the first component to create the x-axis of the rotation
    x = F.normalize(x1, dim=-1)
    # Orthogonalize y1 to x to ensure orthogonality and normalize it
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    # Compute the cross product to get the z-axis
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)


def get_min_max_cameras_distances(poses: list) -> tuple:
    """
    return maximum pose distance from origin

    Args:
        poses (list): list of numpy (4, 4) poses

    Returns:
        min_dist (float): minumum camera distance from origin
        max_dist (float): maximum camera distance from origin
    """
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


# def apply_rotation_3d(
#     points_3d: Union[np.ndarray, torch.Tensor],
#     rot: Union[np.ndarray, torch.Tensor]
# ) -> Union[np.ndarray, torch.Tensor]:
#     """
#     Applies a 3D rotation to a set of points.

#     Args:
#         points_3d (numpy.ndarray or torch.Tensor): A (N, 3) array of 3D points.
#         rot (numpy.ndarray or torch.Tensor): A (3, 3) rotation matrix.

#     Returns:
#         numpy.ndarray or torch.Tensor: A (N, 3) array of rotated 3D points.

#     Raises:
#         ValueError: If the shapes of `points_3d` or `rot` are invalid.
#         TypeError: If the input types are inconsistent (mixing NumPy and PyTorch).
#     """
#     # Check if inputs are from the same library
#     if isinstance(points_3d, np.ndarray) and isinstance(rot, np.ndarray):
#         lib = np
#     elif isinstance(points_3d, torch.Tensor) and isinstance(rot, torch.Tensor):
#         lib = torch
#     else:
#         raise TypeError(
#             "Both `points_3d` and `rot` must be either NumPy arrays or PyTorch tensors."
#         )

#     # Validate input shapes
#     if points_3d.shape[1] != 3:
#         raise ValueError("`points_3d` must have shape (N, 3).")
#     if rot.shape != (3, 3):
#         raise ValueError("`rot` must be a 3x3 rotation matrix.")

#     # Apply rotation
#     rotated_points = lib.matmul(
#         points_3d, rot.T
#     )  # Rotates points, handling (N, 3) @ (3, 3)

#     return rotated_points


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


# def apply_transformation_3d(
#     points_3d: Union[np.ndarray, torch.Tensor],
#     transform: Union[np.ndarray, torch.Tensor],
# ) -> Union[np.ndarray, torch.Tensor]:
#     """
#     Applies a 3D affine transformation to a set of points.

#     Args:
#         points_3d (numpy.ndarray or torch.Tensor): A (N, 3) array of 3D points.
#         transform (numpy.ndarray or torch.Tensor): A (4, 4) affine transformation matrix.

#     Returns:
#         numpy.ndarray or torch.Tensor: A (N, 3) array of transformed 3D points.

#     Raises:
#         ValueError: If the shapes of `points_3d` or `transform` are invalid.
#         TypeError: If the input types are inconsistent (mixing NumPy and PyTorch).
#     """
#     # Check if inputs are from the same library
#     if isinstance(points_3d, np.ndarray) and isinstance(transform, np.ndarray):
#         lib = np
#     elif isinstance(points_3d, torch.Tensor) and isinstance(transform, torch.Tensor):
#         lib = torch
#     else:
#         raise TypeError(
#             "Both `points_3d` and `transform` must be either NumPy arrays or PyTorch tensors."
#         )

#     # Validate input shapes
#     if points_3d.shape[1] != 3:
#         raise ValueError("`points_3d` must have shape (N, 3).")
#     if transform.shape != (4, 4):
#         raise ValueError("`transform` must be a 4x4 transformation matrix.")

#     # Convert points to homogeneous coordinates
#     if lib == torch:
#         ones = lib.ones(
#             (points_3d.shape[0], 1), dtype=points_3d.dtype, device=points_3d.device
#         )
#         augmented_points_3d = lib.concat([points_3d, ones], axis=1)  # (N, 4)
#     elif lib == np:
#         ones = lib.ones((points_3d.shape[0], 1), dtype=points_3d.dtype)
#         augmented_points_3d = lib.hstack([points_3d, ones])  # (N, 4)

#     # Apply transformation
#     transformed_points_3d = lib.matmul(augmented_points_3d, transform.T)  # (N, 4)

#     # Convert back to Euclidean coordinates
#     points_3d = transformed_points_3d[:, :3] / transformed_points_3d[:, 3:4]

#     return points_3d


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


def pad_matrix(
    matrix: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Pads a transformation matrix with a homogeneous bottom row [0, 0, 0, 1].

    Args:
        matrix (np.ndarray or torch.Tensor): A (3, 4) or (N, 3, 4) transformation matrix.

    Returns:
        np.ndarray or torch.Tensor: A (4, 4) or (N, 4, 4) transformation matrix with the bottom row added.

    Raises:
        ValueError: If `matrix` is not a valid 2D or 3D transformation matrix.
    """
    if isinstance(matrix, np.ndarray):
        if matrix.ndim == 2 and matrix.shape == (3, 4):  # Single matrix case
            bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype)
            padded_matrix = np.vstack([matrix, bottom[None, :]])
        elif matrix.ndim == 3 and matrix.shape[1:] == (3, 4):  # Batch case
            bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype)
            bottom = np.tile(bottom, (matrix.shape[0], 1, 1))  # Expand for batch
            padded_matrix = np.concatenate([matrix, bottom[:, None, :]], axis=1)
        else:
            raise ValueError("Invalid matrix shape. Expected (3, 4) or (N, 3, 4).")
    elif isinstance(matrix, torch.Tensor):
        if matrix.ndim == 2 and matrix.shape == (3, 4):  # Single matrix case
            bottom = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=matrix.device, dtype=matrix.dtype
            )
            padded_matrix = torch.cat([matrix, bottom[None, :]], dim=0)
        elif matrix.ndim == 3 and matrix.shape[1:] == (3, 4):  # Batch case
            bottom = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=matrix.device, dtype=matrix.dtype
            )
            bottom = bottom.expand(matrix.shape[0], 1, 4)  # Expand for batch
            padded_matrix = torch.cat([matrix, bottom], dim=1)
        else:
            raise ValueError("Invalid matrix shape. Expected (3, 4) or (N, 3, 4).")
    else:
        raise ValueError(
            "Unsupported matrix type, should be np.ndarray or torch.Tensor."
        )

    return padded_matrix


def unpad_matrix(
    matrix: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Removes the homogeneous bottom row from a padded transformation matrix.

    Args:
        matrix (np.ndarray or torch.Tensor): A (4, 4) or (N, 4, 4) transformation matrix.

    Returns:
        np.ndarray or torch.Tensor: A (3, 4) or (N, 3, 4) transformation matrix.

    Raises:
        ValueError: If `matrix` does not have the correct shape.
    """
    if matrix.ndim == 2:
        if matrix.shape != (4, 4):
            raise ValueError("Invalid matrix shape. Expected (4, 4).")
        return matrix[:3, :]  # Single matrix case
    elif matrix.ndim == 3:
        if matrix.shape[1:] != (4, 4):
            raise ValueError("Invalid batch matrix shape. Expected (N, 4, 4).")
        return matrix[:, :3, :]  # Batch case
    else:
        raise ValueError("Unsupported matrix dimensionality, should be 2D or 3D.")


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
        intrinsics = torch.tensor(
            intrinsics, dtype=torch.float32, device=points_3d_world.device
        )
        c2w = torch.tensor(c2w, dtype=torch.float32, device=points_3d_world.device)
        w2c = torch.inverse(c2w)  # World-to-camera transformation
    elif isinstance(points_3d_world, np.ndarray):
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
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
        intrinsics_inv = torch.tensor(
            intrinsics_inv, dtype=torch.float32, device=points_2d_screen.device
        )
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

    # Transform points from camera space to world space
    points_3d_world = (c2w[:3, :3] @ points_3d_camera.T).T

    # Normalize the direction vectors
    if isinstance(points_3d_world, torch.Tensor):
        rays_d = F.normalize(points_3d_world, dim=-1)
    else:
        rays_d = points_3d_world / np.linalg.norm(
            points_3d_world, axis=-1, keepdims=True
        )

    # Scale direction vectors by depth
    points_3d_world = rays_d * depth[..., None]

    # Add ray origin to scale and translate points
    points_3d_world += rays_o

    return points_3d_world


# def look_at(eye, center, up, forward_positive_z=False):
#     """
#     Compute camera pose from look-at vectors.

#     Args:
#         eye (np.ndarray): (3,) Camera position in world space.
#         center (np.ndarray): (3,) Point to look at in world space.
#         up (np.ndarray): (3,) Up vector.
#         forward_positive_z (bool): If True, forward points to +Z. If False, forward points to -Z.

#     Returns:
#         np.ndarray: (4, 4) Camera pose matrix.
#     """
#     assert eye.shape == (3,)
#     assert center.shape == (3,)
#     assert up.shape == (3,)

#     # Compute forward vector
#     forward = center - eye if forward_positive_z else eye - center
#     forward = forward / np.linalg.norm(forward)

#     # Compute right and up vectors
#     right = np.cross(up, forward)
#     right = right / np.linalg.norm(right)
#     new_up = np.cross(forward, right)
#     new_up = new_up / np.linalg.norm(new_up)

#     # Construct rotation matrix
#     rotation = np.eye(4)
#     rotation[:3, 0] = right
#     rotation[:3, 1] = new_up
#     rotation[:3, 2] = forward if forward_positive_z else -forward

#     # Add translation
#     rotation[:3, 3] = eye

#     return rotation


def look_at(
    camera_origin: np.ndarray, target_point: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Compute camera pose from look at vectors
    args:
        camera_origin (np.ndarray) : (3,) camera position
        target_point (np.ndarray) : (3,) point to look at
        up (np.ndarray) : (3,) up vector
    out:
        pose (np.ndarray) : (4, 4) camera pose
    """

    assert camera_origin.shape == (3,)
    assert target_point.shape == (3,)
    assert up.shape == (3,)

    # get camera frame
    z = camera_origin - target_point
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    # get rotation matrix
    rotation = np.eye(3)
    rotation[:, 0] = x
    rotation[:, 1] = y
    rotation[:, 2] = z

    # add translation
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = camera_origin

    return pose


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
