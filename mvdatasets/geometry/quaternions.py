import torch
import numpy as np


def quats_to_rots(quats):
    # Determine whether the input is a torch tensor or numpy array
    if isinstance(quats, torch.Tensor):
        lib = torch
    elif isinstance(quats, np.ndarray):
        lib = np
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")

    # Extract components of the quaternion
    r = quats[..., 0]
    i = quats[..., 1]
    j = quats[..., 2]
    k = quats[..., 3]

    # Compute scaling factor
    two_s = 2.0 / (quats * quats).sum(-1, keepdims=True)

    # Compute rotation matrix components
    o = lib.stack(
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
        axis=-1 if lib is np else -1,
    )

    # Reshape the output
    return (
        o.reshape(quats.shape[:-1] + (3, 3))
        if lib is np
        else o.view(quats.shape[:-1] + (3, 3))
    )
