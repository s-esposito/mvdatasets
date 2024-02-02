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


def apply_transformation_3d(points_3d, transform):
    """apply linear transformation to points
    args:
        points_3d (N, 3)
        transform (4, 4)
    out: points (N, 3)
    """
    augmented_points_3d = augment_vectors(points_3d)
    homogeneous_points_3d = (transform @ augmented_points_3d.T).T
    augmented_points_3d = homogeneous_points_3d / homogeneous_points_3d[:, 3:]
    points_3d = augmented_points_3d[:, :3]
    return points_3d


def augment_vectors(vectors):
    """concatenate ones to vectors
    args:
        vectors (np.ndarray or torch.tensor) : (N, C)
    out: 
        vectors (np.ndarray or torch.tensor) : (N, C+1)
    """
    
    if torch.is_tensor(vectors):
        return torch.cat(
            [
                vectors,
                torch.ones_like(vectors[:, :1], device=vectors.device)
            ], dim=-1)
    elif isinstance(vectors, np.ndarray):
        return np.concatenate(
            [
                vectors,
                np.ones_like(vectors[:, :1])
            ], axis=-1)
    else:
        raise ValueError("vectors must be torch.tensor or np.ndarray")


def perspective_projection(intrinsics, points_3d):
    """apply perspective projection to points
    args:
        intrinsics (np.array) : (3, 3)
        points_3d (np.array) : (N, 3)
    out: 
        points_2d (np.array) : (N, 2)
    """
    augmented_points_3d = augment_vectors(points_3d)
    K0 = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
    homogeneous_points_2d = (K0 @ augmented_points_3d.T).T
    augmented_points_2d = homogeneous_points_2d / homogeneous_points_2d[:, 2:]
    points_2d = augmented_points_2d[:, :2]
    return points_2d

    
def inv_perspective_projection(intrinsics_inv, points_2d_screen):
    """apply inverse perspective projection to 2d points
    args:
        intrinsics_inv (np.ndarray or torch.tensor) : (3, 3)
        points_2d_screen (np.ndarray or torch.tensor) : (N, 2) point in screen coordinates (u,v)
    out: 
        points_3d (np.ndarray or torch.tensor) : (N, 3)
    """
    
    assert intrinsics_inv.shape == (3, 3)
    assert points_2d_screen.shape[-1] == 2
    
    augmented_points_2d_screen = augment_vectors(points_2d_screen)
    augmented_points_3d_camera = (intrinsics_inv @ augmented_points_2d_screen.T).T
    
    return augmented_points_3d_camera


def project_points_3d_to_2d(camera, points_3d):
    """Project 3D points to 2D
    args:
        points_3d (np.ndarray) : (N, 3) points in world space
        c2w (np.ndarray) : (4, 4) camera pose in world space
        intrinsics (np.ndarray) : (3, 3) camera intrinsics
    out:
        points_2d (np.ndarray) : (N, 2) points in screen space
    """

    # get camera data
    intrinsics = camera.get_intrinsics()
    c2w = camera.get_pose()

    # get world to camera transformation
    w2c = np.linalg.inv(c2w)
    
    # transform points in world space to camera space
    points_3d_c = apply_transformation_3d(points_3d, w2c)

    # convert homogeneous coordinates to 2d coordinates
    points_2d_s = perspective_projection(intrinsics, points_3d_c)

    return points_2d_s


def camera_to_points_3d_distance(camera, points_3d):
    
    # get camera data
    c2w = camera.get_pose()
    
    # get world to camera transformation
    w2c = np.linalg.inv(c2w)

    # transform points in world space to camera space
    points_3d_c = apply_transformation_3d(points_3d, w2c)
    
    points_3d_norm = np.linalg.norm(points_3d_c, axis=-1)
    
    return points_3d_norm
    
    
def look_at(eye, center, up):
    """Compute camera pose from look at vectors
    args:
        eye (np.ndarray) : (3,) camera position
        center (np.ndarray) : (3,) point to look at
        up (np.ndarray) : (3,) up vector
    out:
        pose (np.ndarray) : (4, 4) camera pose
    """
    
    assert eye.shape == (3,)
    assert center.shape == (3,)
    assert up.shape == (3,)
    
    # get camera frame
    z = eye - center
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    
    # get rotation matrix
    rotation = np.eye(3)
    rotation[:3, 0] = x
    rotation[:3, 1] = y
    rotation[:3, 2] = z
    
    return rotation


############################################
# from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L523


def qvec2rotmat(qvec):
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


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
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


############################################