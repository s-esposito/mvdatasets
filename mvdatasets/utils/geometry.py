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


def transform_points_3d(points, transform):
    """apply transform to points
    args:
        points (N, 3)
        transform (4, 4)
    out: points (N, 3)
    """
    return np.dot(transform[:3, :3], points.T).T + transform[:3, 3]


def project_points_3d_to_2d(camera, points_3d):
    """Project 3D points to 2D points
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

    # view transformation
    points_3d_camera = transform_points_3d(points_3d, w2c)

    # perspective transformation
    points_2d_homogeneous = (intrinsics @ points_3d_camera.T).T

    # convert to 2D coordinates by dividing by the last column (homogeneous coordinate)
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]

    # filter out points outside image range
    points_2d = points_2d[points_2d[:, 0] > 0]
    points_2d = points_2d[points_2d[:, 1] > 0]
    points_2d = points_2d[points_2d[:, 0] < camera.width]
    points_2d = points_2d[points_2d[:, 1] < camera.height]

    # flip image plane y axis
    points_2d[:, 1] = camera.height - points_2d[:, 1]

    return points_2d
