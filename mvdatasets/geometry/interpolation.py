import numpy as np
import torch


def lerp(x0, x1, h):
    h = torch.clip(h, 0, 1)
    return (1 - h) * x0 + h * x1


def slerp(q1, q2, t):
    """
    Performs spherical linear interpolation (SLERP) between batches of quaternions q1 and q2.

    Args:
        q1: Tensor of shape (N, T, 4) representing the first batch of quaternions.
        q2: Tensor of shape (N, T, 4) representing the second batch of quaternions.
        t: A scalar or tensor of shape (N, T) or (N, 1) representing the interpolation factors (0 <= t <= 1).

    Returns:
        Interpolated quaternion tensor of shape (N, T, 4).
    """

    # Convert to torch tensors if not already
    is_numpy = False
    if isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray):
        q1 = torch.tensor(q1, dtype=torch.float32)
        q2 = torch.tensor(q2, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        is_numpy = True

    # Ensure quaternions are normalized
    q1 = q1 / torch.linalg.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.linalg.norm(q2, dim=-1, keepdim=True)

    # Compute the dot product (cosine of the angle between the quaternions)
    dot_product = torch.sum(q1 * q2, dim=-1, keepdim=True)  # Shape: (N, T, 1)

    # If the dot product is negative, negate q2 to take the shortest path
    mask = (dot_product < 0.0).squeeze(-1)
    q2[mask] = -q2[mask]

    dot_product = torch.abs(dot_product)

    # Clamp dot product to avoid numerical issues (should be in [-1, 1])
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the angle between the two quaternions
    theta_0 = torch.acos(dot_product)  # Shape: (N, T, 1)

    # If the angle is very small, fall back to LERP (Linear interpolation)
    small_angle_mask = (theta_0 < 1e-6).squeeze(-1)
    if small_angle_mask.any():
        q_lerp = (1.0 - t[..., None]) * q1 + t[..., None] * q2
        q_lerp = q_lerp / torch.linalg.norm(q_lerp, dim=-1, keepdim=True)
        q1[small_angle_mask] = q_lerp[small_angle_mask]
        return q1

    # Compute sin(theta_0)
    sin_theta_0 = torch.sin(theta_0)  # Shape: (N, T, 1)

    # Compute the two interpolation terms
    s1 = torch.sin((1.0 - t)[..., None] * theta_0) / sin_theta_0  # Shape: (N, T, 1)
    s2 = torch.sin(t[..., None] * theta_0) / sin_theta_0  # Shape: (N, T, 1)

    # Compute the interpolated quaternion
    q_slerp = s1 * q1 + s2 * q2  # Shape: (N, T, 4)

    # Return the normalized interpolated quaternion
    q_slerp = q_slerp / torch.linalg.norm(q_slerp, dim=-1, keepdim=True)

    # Convert back to numpy if necessary
    if is_numpy:
        q_slerp = q_slerp.cpu().numpy()

    return q_slerp
