import torch

# from nerfstudio
import matplotlib.pyplot as plt
import numpy as np


def get_min_max_cameras_distances(poses):
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


def opengl_matrix_world_from_w2c(w2c):
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


def opengl_projection_matrix_from_intrinsics(K, width, height, near, far):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    projection_matrix = np.array([
        [2.0 * fx / width, 0.0, -(2.0 * cx / width - 1.0), 0.0],
        [0.0, 2.0 * fy / height, -(2.0 * cy / height - 1.0), 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=np.float32)

    # Transpose to column-major order for OpenGL
    return projection_matrix


def is_valid_pose(c2w):
    if not isinstance(c2w, np.ndarray):
        return False
    if c2w.shape != (4, 4):
        return False
    if not np.allclose(c2w[3], [0, 0, 0, 1]):
        return False
    # Additional checks if needed, e.g., for the upper 3x3 block being a rotation matrix.
    return True


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
    # print("points_3d", points_3d)
    # print("transform", transform)
    augmented_points_3d = euclidean_to_augmented(points_3d)
    homogeneous_points_3d = (transform @ augmented_points_3d.T).T
    # augmented_points_3d = homogeneous_points_3d / homogeneous_points_3d[:, 3:]
    # points_3d = augmented_points_3d[:, :3]
    augmented_points_3d = homogeneous_to_augmented(homogeneous_points_3d)
    points_3d = augmented_to_euclidean(augmented_points_3d)
    return points_3d


def pad_matrix(matrix):
    """Pad matrix with a homogeneous bottom row [0,0,0,1]."""
    if isinstance(matrix, np.ndarray):
        bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype)
        padded_matrix = np.vstack([matrix, bottom[None, :]])
    elif isinstance(matrix, torch.Tensor):
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=matrix.device, dtype=matrix.dtype)
        padded_matrix = torch.cat([matrix, bottom[None, :]], dim=0)
    else:
        raise ValueError('Unsupported matrix type, should be np.ndarray or torch.Tensor.')

    return padded_matrix


def unpad_matrix(matrix):
    """Remove the homogeneous bottom row from pose matrix."""
    return matrix[:3, :4]


def euclidean_to_augmented(vectors):
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
                torch.ones_like(vectors[..., :1], device=vectors.device)
            ], dim=-1)
    elif isinstance(vectors, np.ndarray):
        return np.concatenate(
            [
                vectors,
                np.ones_like(vectors[..., :1])
            ], axis=-1)
    else:
        raise ValueError("vectors must be torch.tensor or np.ndarray")


def homogeneous_to_augmented(vectors):
    """convert homogeneous coordinates to augmented coordinates
    args:
        vectors (np.ndarray or torch.tensor) : (N, C+1)
    out: 
        vectors (np.ndarray or torch.tensor) : (N, C)
    """
    return vectors / vectors[..., -1:]


def augmented_to_euclidean(vectors):
    """convert augmented coordinates to euclidean coordinates
    args:
        vectors (np.ndarray or torch.tensor) : (N, C+1)
    out: 
        vectors (np.ndarray or torch.tensor) : (N, C)
    """
    return vectors[..., :-1]


def perspective_projection(intrinsics, points_3d):
    """apply perspective projection to points
    args:
        intrinsics (np.array) : (3, 3)
        points_3d (np.array) : (N, 3)
    out: 
        points_2d (np.array) : (N, 2)
    """
    augmented_points_3d = euclidean_to_augmented(points_3d)
    if isinstance(intrinsics, torch.Tensor):
        K0 = torch.concatenate([intrinsics, torch.zeros((3, 1), device=intrinsics.device)], dim=1)
    elif isinstance(intrinsics, np.ndarray):
        K0 = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
    homogeneous_points_2d = (K0 @ augmented_points_3d.T).T
    # augmented_points_2d = homogeneous_points_2d / homogeneous_points_2d[:, 2:]
    # points_2d = augmented_points_2d[:, :2]
    augmented_points_2d = homogeneous_to_augmented(homogeneous_points_2d)
    points_2d = augmented_to_euclidean(augmented_points_2d)
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
    
    augmented_points_2d_screen = euclidean_to_augmented(points_2d_screen)
    augmented_points_3d_camera = (intrinsics_inv @ augmented_points_2d_screen.T).T
    
    return augmented_points_3d_camera


def project_points_3d_to_2d_from_intrinsics_and_c2w(intrinsics, c2w, points_3d):
    
    if isinstance(points_3d, torch.Tensor):
        # convert data to torch.tensor
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=points_3d.device)
        c2w = torch.tensor(c2w, dtype=torch.float32, device=points_3d.device)
        # get world to camera transformation
        w2c = torch.inverse(c2w)
    elif isinstance(points_3d, np.ndarray):
        # get world to camera transformation
        w2c = np.linalg.inv(c2w)
    else:
        raise ValueError("points_3d must be torch.tensor or np.ndarray")
    
    # transform points in world space to camera space
    points_3d_c = apply_transformation_3d(points_3d, w2c)

    # convert homogeneous coordinates to 2d coordinates
    points_2d_s = perspective_projection(intrinsics, points_3d_c)

    return points_2d_s


def project_points_3d_to_2d(camera, points_3d):
    """Project 3D points to 2D
    args:
        points_3d (np.ndarray) : (N, 3) points in world space
        c2w (np.ndarray) : (4, 4) camera pose in world space
        intrinsics (np.ndarray) : (3, 3) camera intrinsics
    out:
        points_2d (np.ndarray or torch.tensor) : (N, 2) points in screen space
    """

    # get camera data
    intrinsics = camera.get_intrinsics()
    c2w = camera.get_pose()
    
    return project_points_3d_to_2d_from_intrinsics_and_c2w(intrinsics, c2w, points_3d)


def unproject_points_2d_to_3d_from_intrinsics_and_c2w(intrinsics, c2w, points_2d_s, depth):
    
    # convert data to torch.tensor
    if isinstance(points_2d_s, np.ndarray):
        inv_intrinsics = np.linalg.inv(intrinsics)
    elif isinstance(points_2d_s, torch.Tensor):
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=points_2d_s.device)
        c2w = torch.tensor(c2w, dtype=torch.float32, device=points_2d_s.device)
        # inverse intrinsics for back-projection
        inv_intrinsics = torch.inverse(intrinsics)
    else:
        print("[bold red]ERROR[/bold red]: points_2d_s must be torch.tensor or np.ndarray")
        exit(1)
    
    # ray origin is just the camera center
    rays_o = c2w[:3, -1].unsqueeze(0).expand(points_2d_s.shape[0], -1)

    # pixels have height, width order
    points_3d_c = inv_perspective_projection(
        inv_intrinsics,
        points_2d_s[:, [1, 0]]  # pixels have h, w order but we need x, y
    )
    # points_3d_unprojected have all z=1
    
    # rotate points to world space
    points_3d_w = (c2w[:3, :3] @ points_3d_c.T).T
    
    # normalize rays
    rays_d = torch.nn.functional.normalize(points_3d_w, dim=-1)

    # scale rays by depth
    points_3d_w = rays_d * depth
    
    # add ray origin to points
    points_3d_w += rays_o

    # # points_3d_c = inv_perspective_projection(inv_intrinsics, points_2d_s)
    # # points_3d_c *= depth
    
    # ray_d = inv_perspective_projection(inv_intrinsics, points_2d_s)
    
    # # normalize ray direction
    # if torch.is_tensor(ray_d):
    #     ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
    # elif isinstance(ray_d, np.ndarray):
    #     ray_d = ray_d / np.linalg.norm(ray_d, axis=-1, keepdims=True)
    # else:
    #     raise ValueError("ray_d must be torch.tensor or np.ndarray")
        
    # points_3d_c = ray_d * depth

    # # plot points_2d_s and points_depth
    # plt.figure()
    # plt.scatter(points_3d_c[:, 0], points_3d_c[:, 1], c=np.log(depth[:, 0] + 1), cmap='viridis')
    # # upper left origin
    # plt.gca().invert_yaxis()
    # # axis equal
    # plt.colorbar()
    # plt.axis('equal')
    # plt.show()
    
    # Transform points from camera space to world space
    # points_3d_w = apply_transformation_3d(points_3d_c, c2w)
    # points_3d_w = points_3d_c

    return points_3d_w


def unproject_points_2d_to_3d(camera, points_2d_s, depth):
    """Unproject 2D points to 3D
    args:
        points_2d (np.ndarray or torch.tensor) : (N, 2) points in screen space
        depth (np.ndarray or torch.tensor): (N, 1) depth values for each point
        camera (Camera): Camera object with intrinsics and pose
    out:
        points_3d (np.ndarray or torch.tensor) : (N, 3) points in world space
    """

    # get camera data
    intrinsics = camera.get_intrinsics()
    c2w = camera.get_pose()

    return unproject_points_2d_to_3d_from_intrinsics_and_c2w(intrinsics, c2w, points_2d_s, depth)


def camera_to_points_3d_distance(camera, points_3d):
    
    # get camera data
    c2w = camera.get_pose()
    
    if isinstance(points_3d, torch.Tensor):
        # convert data to torch.tensor
        c2w = torch.tensor(c2w, dtype=torch.float32, device=points_3d.device)
        # get world to camera transformation
        w2c = torch.inverse(c2w)
    elif isinstance(points_3d, np.ndarray):
        # get world to camera transformation
        w2c = np.linalg.inv(c2w)
    else:
        raise ValueError("points_3d must be torch.tensor or np.ndarray")

    # transform points in world space to camera space
    points_3d_c = apply_transformation_3d(points_3d, w2c)
    
    if isinstance(points_3d, torch.Tensor):
        points_3d_norm = torch.norm(points_3d_c, dim=-1)
    elif isinstance(points_3d, np.ndarray):
        points_3d_norm = np.linalg.norm(points_3d_c, axis=-1)
    else:
        raise ValueError("points_3d must be torch.tensor or np.ndarray")
    
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


####################################################
# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/utils/colmap_parsing_utils.py#L454

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