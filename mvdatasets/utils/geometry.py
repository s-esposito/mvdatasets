import torch

# from nerfstudio

import numpy as np


def rotation_matrix(a, b):
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


def deg2rad(deg):
    return deg * np.pi / 180


def scale_3d(scale):
    return np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])


def rot_x_3d(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rot_y_3d(theta):
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rot_z_3d(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def pose_local_rotation(pose, rotation):
    """
    Local rotation of the pose frame by rotation matrix
    Args:
        pose (4, 4)
        rotation (3, 3)
    """
    rotation_transform = np.eye(4)
    rotation_transform[:3, :3] = rotation
    return pose @ rotation_transform


def pose_global_rotation(pose, rotation):
    """
    Global rotation of the pose frame by rotation matrix
    Args:
        pose (4, 4)
        rotation (3, 3)
    """
    rotation_transform = np.eye(4)
    rotation_transform[:3, :3] = rotation
    return rotation_transform @ pose


def linear_transformation_2d(points, transform):
    """apply linear transformation to points
    args:
        points (N, 2)
        transform (3, 3)
    out: points (N, 2)
    """
    return np.dot(transform[:2, :2], points.T).T + transform[:2, 2]


def linear_transformation_3d(points, transform):
    """apply linear transformation to points
    args:
        points (N, 3)
        transform (4, 4)
    out: points (N, 3)
    """
    return np.dot(transform[:3, :3], points.T).T + transform[:3, 3]


def perspective_projection(intrinsics, points_3d):
    """apply perspective projection to points
    args:
        intrinsics (3, 3)
        points_3d (N, 3)
    out: points_2d (N, 2)
    """
    points_2d_homogeneous = (intrinsics @ points_3d.T).T
    return points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]


def concat_ones(points):
    """concatenate ones to points
    args:
        points (np.ndarray or torch.tensor) : (N, C)
    out: 
        points (np.ndarray or torch.tensor) : (N, C+1)
    """
    
    if torch.is_tensor(points):
        return torch.cat([points, torch.ones_like(points[:, :1], device=points.device)], dim=-1)
    elif isinstance(points, np.ndarray):
        return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
    else:
        raise ValueError("points must be either torch.tensor or np.ndarray")
    
    
def inv_perspective_projection(intrinsics_inv, points_2d):
    """apply inverse perspective projection to points
    args:
        intrinsics (np.ndarray or torch.tensor) : (3, 3)
        points_2d (np.ndarray or torch.tensor) : (N, 2) -> (x, y)
    out: 
        points_3d (np.ndarray or torch.tensor) : (N, 3)
    """
    
    points_2d_augmented = concat_ones(points_2d)
    points_3d_unprojected = (intrinsics_inv @ points_2d_augmented.T).T
    
    return points_3d_unprojected


def project_points_3d_to_2d(camera, points_3d):
    """Project 3D points to 2D
    args:
        points_3d (np.ndarray) : (N, 3)
        c2w (np.ndarray) : (4, 4)
        intrinsics (np.ndarray) : (3, 3)
    out:
        points_2d (np.ndarray) : (N, 2)
    """

    # get camera data
    intrinsics = camera.get_intrinsics()
    c2w = camera.get_pose()

    # get world to camera transformation
    w2c = np.linalg.inv(c2w)

    # transform points in world space to camera space
    points_3d_camera = linear_transformation_3d(points_3d, w2c)

    # convert homogeneous coordinates to 2d coordinates
    points_2d = perspective_projection(intrinsics, points_3d_camera)

    return points_2d
