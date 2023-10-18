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
    return np.eye(3) + skew_sym_mat + np.dot(skew_sym_mat, skew_sym_mat) * ((1 - c) / (s**2 + 1e-8))


def deg2rad(deg):
    return deg * np.pi / 180


def rot_x_3d(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def rot_y_3d(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def rot_z_3d(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


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


def project_points_3d_to_2d(points, camera):
    """Project 3D points to 2D points
    args: points (N, 3)
    projection_matrix (3, 4)
    out: points (N, 2)
    """

    # transform the points from the world coordinate system to the camera coordinate system using the inverse of the camera pose matrix
    transformed_points = transform_points_3d(points, np.linalg.inv(camera.get_pose()))

    # Assuming camera_intrinsics is a 3x3 numpy array representing the intrinsic matrix
    projected_points = np.dot(camera.intrinsics, transformed_points.T).T
    # Normalize the coordinates
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    return projected_points[:, :2]
