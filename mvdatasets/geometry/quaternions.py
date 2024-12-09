import torch
import numpy as np
from typing import Union


def quat_multiply(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quat_invert(q):
    """
    Inverts a batch of quaternions q.
    q: (N, 4) batch of quaternions
    Returns: (N, 4) batch of inverted quaternion
    """

    # Ensure quaternions are normalized
    q = q / torch.linalg.norm(q, dim=1, keepdim=True)

    w = q[:, 0]
    x = -q[:, 1]
    y = -q[:, 2]
    z = -q[:, 3]
    inv_q = torch.stack([w, x, y, z], dim=1)  # (N, 4)
    # check if inv_q has 1 dimension
    if len(inv_q.shape) == 1:
        inv_q = inv_q.unsqueeze(0)
    # Normalize the inverted quaternion
    inv_q = inv_q / torch.linalg.norm(inv_q, dim=1, keepdim=True)
    return inv_q  # (N, 4)


def make_quaternion_deg(deg_x, deg_y, deg_z, dtype=torch.float32, device="cpu"):
    return make_quaternion_rad(
        np.radians(deg_x),
        np.radians(deg_y),
        np.radians(deg_z),
        dtype=dtype,
        device=device,
    )


def make_quaternion_rad(rad_x, rad_y, rad_z, dtype=torch.float32, device="cpu"):
    q = torch.tensor(
        [
            np.cos(rad_x / 2) * np.cos(rad_y / 2) * np.cos(rad_z / 2)
            + np.sin(rad_x / 2) * np.sin(rad_y / 2) * np.sin(rad_z / 2),
            np.sin(rad_x / 2) * np.cos(rad_y / 2) * np.cos(rad_z / 2)
            - np.cos(rad_x / 2) * np.sin(rad_y / 2) * np.sin(rad_z / 2),
            np.cos(rad_x / 2) * np.sin(rad_y / 2) * np.cos(rad_z / 2)
            + np.sin(rad_x / 2) * np.cos(rad_y / 2) * np.sin(rad_z / 2),
            np.cos(rad_x / 2) * np.cos(rad_y / 2) * np.sin(rad_z / 2)
            - np.sin(rad_x / 2) * np.sin(rad_y / 2) * np.cos(rad_z / 2),
        ],
        dtype=dtype,
        device=device,
    )
    # Normalize the quaternion
    q = q / torch.linalg.norm(q)
    return q


def rot_to_quat(rots: torch.tensor) -> torch.tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if rots.size(-1) != 3 or rots.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {rots.shape}.")

    batch_dim = rots.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rots.reshape(batch_dim + (9,)), dim=-1
    )

    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def angular_distance(q1, q2):
    """compute row-wise angular distance between two batches of quaternions

    Args:
        q1 (torch.tensor): (N, 4) batch of quaternions
        q2 (torch.tensor): (N, 4) batch of quaternions
    Output:
        theta (torch.tensor): (N,) batch of angular distances
    """

    # Ensure quaternions are normalized
    q1 = q1 / torch.linalg.norm(q1, dim=1, keepdim=True)
    q2 = q2 / torch.linalg.norm(q2, dim=1, keepdim=True)

    # theta = 2 * arccos(|dot(q1, q2)|)
    dot_product = torch.sum(q1 * q2, dim=-1)
    # theta = 2 * torch.acos(torch.abs(dot_product))
    # simplify to improve efficiency
    theta = 1 - torch.abs(dot_product)

    return theta


def quats_to_rots(
    quats: Union[torch.tensor, np.ndarray]
) -> Union[torch.tensor, np.ndarray]:
    # Determine whether the input is a torch tensor or numpy array
    if isinstance(quats, torch.Tensor):
        r, i, j, k = torch.unbind(quats, -1)
        two_s = 2.0 / (quats * quats).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.view(quats.shape[:-1] + (3, 3))
    elif isinstance(quats, np.ndarray):
        r, i, j, k = np.split(quats, 4, axis=-1)
        two_s = 2.0 / (quats * quats).sum(axis=-1, keepdims=True)
        o = np.concatenate(
            [
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ],
            axis=-1,
        )
        shape = quats.shape[:-1] + (3, 3)
        return o.reshape(shape)
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")


# def quats_to_rots(quats):
#     # Determine whether the input is a torch tensor or numpy array
#     if isinstance(quats, torch.Tensor):
#         lib = torch
#     elif isinstance(quats, np.ndarray):
#         lib = np
#     else:
#         raise TypeError("Input must be a torch.Tensor or np.ndarray")

#     # Extract components of the quaternion
#     r = quats[..., 0]
#     i = quats[..., 1]
#     j = quats[..., 2]
#     k = quats[..., 3]

#     # Compute scaling factor
#     two_s = 2.0 / (quats * quats).sum(-1, keepdims=True)

#     # Compute rotation matrix components
#     o = lib.stack(
#         (
#             1 - two_s * (j * j + k * k),
#             two_s * (i * j - k * r),
#             two_s * (i * k + j * r),
#             two_s * (i * j + k * r),
#             1 - two_s * (i * i + k * k),
#             two_s * (j * k - i * r),
#             two_s * (i * k - j * r),
#             two_s * (j * k + i * r),
#             1 - two_s * (i * i + j * j),
#         ),
#         axis=-1 if lib is np else -1,
#     )

#     # Reshape the output
#     return (
#         o.reshape(quats.shape[:-1] + (3, 3))
#         if lib is np
#         else o.view(quats.shape[:-1] + (3, 3))
#     )
