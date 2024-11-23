import torch
import numpy as np
from typing import Tuple, Union

from mvdatasets.geometry.common import (
    local_inv_perspective_projection,
    apply_rotation_3d,
)
from mvdatasets.utils.images import image_uint8_to_float32


def get_pixels(height: int, width: int, device: str = "cpu") -> torch.Tensor:
    """returns all image pixels coords

    out:
        pixels (torch.Tensor): dtype int32, shape (W, H, 2), values in [0, W-1], [0, H-1]
    """

    pixels_x, pixels_y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="ij",
    )
    pixels = torch.stack([pixels_x, pixels_y], dim=-1).int()

    return pixels


def get_random_pixels(
    height: int, width: int, nr_pixels: int, device: str = "cpu"
) -> torch.Tensor:
    """given a number or pixels, return random pixels

    out:
        pixels (torch.Tensor, int): (N, 2) with values in [0, W-1], [0, H-1]
    """
    # sample nr_pixels random pixels
    pixels = torch.rand(nr_pixels, 2, device=device)
    pixels[:, 0] *= width
    pixels[:, 1] *= height
    pixels = pixels.int()
    return pixels


def get_random_pixels_from_error_map(
    error_map: torch.Tensor, nr_pixels: int, device: str = "cpu"
) -> torch.Tensor:
    """given a number of pixels and an error map, sample pixels with error map as probability

    Args:
        error_map (torch.Tensor): (H, W, 1) with values in [0, 1]
        height (int): frame height
        width (int): frame width
        nr_pixels (int): number of pixels to sample
        device (str, optional): Defaults to "cpu".
    """

    # check device
    if error_map.device != device:
        error_map = error_map.to(device)

    height, width = error_map.shape[:2]

    # convert error map to probabilities
    probabilities = error_map.view(-1)
    # normaliza probabilities to ensure they sum up to 1
    probabilities = probabilities / probabilities.sum()
    # sample pixel indices based on probabilities
    pixels_1d = torch.multinomial(probabilities, nr_pixels, replacement=False)

    # convert 1d indices to 2d indices and convert to int
    pixels = torch.stack([pixels_1d // width, pixels_1d % width], dim=1).int()

    # invert y, x to x, y
    pixels = pixels[:, [1, 0]]

    return pixels


def get_pixels_centers(pixels: torch.Tensor) -> torch.Tensor:
    """return the center of each pixel

    args:
        pixels (torch.Tensor): (N, 2) list of pixels
    out:
        pixels_centers (torch.Tensor): (N, 2) list of pixels centers
    """

    points_2d_screen = pixels.float()  # cast to float32
    points_2d_screen = points_2d_screen + 0.5  # pixels centers

    return points_2d_screen


def points_2d_screen_to_pixels(points_2d_screen: torch.Tensor) -> torch.Tensor:
    """convert 2d points on the image plane to pixels
    args:
        points_2d_screen (torch.Tensor): (N, 2) list of pixels centers (in screen space)
    """
    return points_2d_screen.int()  # cast to int32 (floor)


def jitter_points(points: torch.Tensor) -> torch.Tensor:
    """apply noise to points

    Args:
        points (torch.Tensor): (..., 2) list of pixels centers (in screen space)
    Out:
        jittered_pixels (torch.Tensor): (..., 2) list of pixels
    """

    assert points.dtype == torch.float32, "points must be float32"

    # # sample offsets from gaussian distribution
    # std = 0.16
    # offsets = torch.normal(
    #     mean=0.0, std=std, size=jittered_points.shape, device=points.device
    # )
    # clamp offsets to [-0.5 + eps, 0.5 - eps]

    # uniformlu sampled offsets
    offsets = torch.rand_like(points, device=points.device)
    offsets -= 0.5  # [-0.5, 0.5]
    eps = 1e-6
    offsets = torch.clamp(offsets, -0.5 + eps, 0.5 - eps)
    return points + offsets


def get_points_2d_screen_from_pixels(pixels: torch.Tensor, jitter_pixels: bool = False):
    """convert pixels to 2d points on the image plane

    args:
        pixels (torch.Tensor): (W, H, 2) or (N, 2) list of pixels
        jitter_pixels (bool): whether to jitter pixels
    out:
        points_2d_screen (torch.Tensor): (N, 2) list of pixels centers (in screen space)
    """
    assert pixels.dtype == torch.int32, "pixels must be int32"

    # get pixels as 3d points on a plane at z=-1 (in camera space)
    points_2d_screen = get_pixels_centers(pixels)
    points_2d_screen = points_2d_screen.reshape(-1, 2)
    if jitter_pixels:
        points_2d_screen = jitter_points(points_2d_screen)

    return points_2d_screen  # (N, 2)


def get_rays_per_points_2d_screen(
    c2w: torch.Tensor, intrinsics_inv: torch.Tensor, points_2d_screen: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """given a list of pixels, return rays origins and directions

    args:
        c2w (torch.Tensor): (N, 4, 4) or (4, 4)
        intrinsics_inv (torch.Tensor): (N, 3, 3) or (3, 3)
        points_2d_screen (torch.Tensor, float): (N, 2) with values in [0, W-1], [0, H-1]

    out:
        rays_o (torch.Tensor): (N, 3)
        rays_d (torch.Tensor): (N, 3)
    """

    # check input shapes
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    elif c2w.ndim == 3:
        pass
    else:
        raise ValueError(f"c2w: {c2w.shape} must be (4, 4) or (N, 4, 4)")

    if c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w: {c2w.shape} must be (4, 4) or (N, 4, 4)")

    if intrinsics_inv.ndim == 2:
        intrinsics_inv = intrinsics_inv.unsqueeze(0)
    elif intrinsics_inv.ndim == 3:
        pass
    else:
        raise ValueError(
            f"intrinsics_inv: {intrinsics_inv} must be (N, 3, 3) or (3, 3)"
        )

    if intrinsics_inv.shape[1:] != (3, 3):
        raise ValueError(
            f"intrinsics_inv: {intrinsics_inv} must be (N, 3, 3) or (3, 3)"
        )

    if points_2d_screen.ndim != 2 or points_2d_screen.shape[1] != 2:
        raise ValueError(f"points_2d_screen: {points_2d_screen.shape} must be (N, 2)")
    if c2w.shape[0] != points_2d_screen.shape[0] and c2w.shape[0] != 1:
        raise ValueError(
            f"input shapes do not match: c2w: {c2w.shape} and points_2d_screen: {points_2d_screen.shape}"
        )
    if (
        intrinsics_inv.shape[0] != points_2d_screen.shape[0]
        and intrinsics_inv.shape[0] != 1
    ):
        raise ValueError(
            f"input shapes do not match: intrinsics_inv: {intrinsics_inv.shape} and points_2d_screen: {points_2d_screen.shape}"
        )

    # ray origin are the cameras centers
    if c2w.shape[0] == points_2d_screen.shape[0]:
        rays_o = c2w[:, :3, -1]
    else:
        rays_o = c2w[0, :3, -1].repeat(points_2d_screen.shape[0], 1)

    # unproject points to 3d camera space
    points_3d_camera = local_inv_perspective_projection(
        intrinsics_inv,
        points_2d_screen,
    )  # (N, 3)
    # points_3d_unprojected have all z=1

    # rotate points with c2w rotation
    rot = c2w[:, :3, :3]
    points_3d_rotated = apply_rotation_3d(points_3d_camera, rot)  # (N, 3)

    # normalize rays
    rays_d = torch.nn.functional.normalize(points_3d_rotated, dim=-1)  # (N, 3)

    return rays_o, rays_d


def get_data_per_pixels(
    pixels: torch.Tensor,
    cameras_idx: Union[torch.Tensor, np.ndarray] = None,
    frames_idx: Union[torch.Tensor, np.ndarray] = None,
    data_dict: dict = {},
    verbose: bool = False,
):
    """given a list of pixels and a list of frames, return rgb and mask values at pixels

    args:
        pixels (torch.Tensor, int32): (N, 2) with values in [0, W-1], [0, H-1]
        cameras_idx (optional, torch.Tensor, int): (N) camera indices
        frames_idx (optional, torch.Tensor, int): (N) frame indices.
        data_dict (dict, uint8):
            rgbs (optional, torch.Tensor or np.array, uint8): (N, T, H, W, 3) or (H, W, 3) None
            masks (optional, torch.Tensor or np.array, uint8): (N, T, H, W, 1) or (H, W, 1) or None

    out:
        vals (dict):
            rgb (optional, torch.Tensor or np.array, float): (N, 3)
            mask (optional, torch.Tensor or np.array, float): (N, 1)
    """

    assert pixels.ndim == 2, "points_2d_screen must be (N, 2)"
    assert pixels.shape[1] == 2, "points_2d_screen must be (N, 2)"
    assert pixels.dtype == torch.int32, "pixels must be int32"
    if cameras_idx is not None:
        assert cameras_idx.ndim == 1, f"cameras_idx: {cameras_idx.shape[0]} must be 1D"
        if isinstance(cameras_idx, np.ndarray):
            assert cameras_idx.dtype == np.int32, "cameras_idx must be int32"
        if isinstance(cameras_idx, torch.Tensor):
            assert cameras_idx.dtype == torch.int32, "cameras_idx must be int32"
        assert (
            cameras_idx.shape[0] == pixels.shape[0]
        ), f"cameras_idx: {cameras_idx.shape[0]} must have the same length as pixels: {pixels.shape[0]}"
    if frames_idx is not None:
        assert frames_idx.ndim == 1, f"frames_idx must: {frames_idx} be 1D"
        if isinstance(frames_idx, np.ndarray):
            assert frames_idx.dtype == np.int32, "frames_idx must be int32"
        if isinstance(frames_idx, torch.Tensor):
            assert frames_idx.dtype == torch.int32, "frames_idx must be int32"
        assert (
            frames_idx.shape[0] == pixels.shape[0]
        ), f"frames_idx: {frames_idx.shape[0]} must have the same length as pixels: {pixels.shape[0]}"

    # invert w, h to h, w
    i, j = pixels[:, 1], pixels[:, 0]

    # prepare output
    vals = {}

    for key, val in data_dict.items():
        if val is not None:
            if val.ndim == 5:
                if cameras_idx is None:
                    raise ValueError(
                        f"cameras_idx must be provided for data {key} with shape {val.shape}"
                    )
                vals[key] = val[cameras_idx, frames_idx, i, j]  # (N, C)
            else:
                vals[key] = val[frames_idx, i, j]  # (N, C)
            vals[key] = image_uint8_to_float32(vals[key])
        else:
            vals[key] = None

    if verbose:
        for key, val in vals.items():
            if val is not None:
                if isinstance(val, torch.Tensor):
                    print(
                        "torch.Tensor",
                        key,
                        val.shape,
                        val.dtype,
                        val.min().item(),
                        val.max().item(),
                        val.device,
                    )
                elif isinstance(val, np.ndarray):
                    print("np.ndarray", key, val.shape, val.dtype, val.min(), val.max())
            else:
                print(key, val)

    return vals


def get_data_per_points_2d_screen(
    points_2d_screen: torch.Tensor,
    cameras_idx: Union[torch.Tensor, np.ndarray] = None,
    frames_idx: Union[torch.Tensor, np.ndarray] = None,
    data_dict: dict = {},
):
    """given a list of pixels and a list of frames, return rgb and mask values at pixels

    args:
        points_2d_screen (torch.Tensor, float): (N, 2) with values in [0, W-1], [0, H-1]
        cameras_idx (optional, torch.Tensor or np.ndarray): (N) camera indices
        frames_idx (optional, torch.Tensor or np.ndarray): (N) frame indices
        data_dict (dict, uint8):
            rgbs (torch.Tensor, uint8): (H, W, 3)
            mask (torch.Tensor, uint8): (H, W, 1)

    out:
        vals (dict):
            rgb (optional, torch.Tensor, float): (N, 3)
            mask (optional, torch.Tensor, float): (N, 1)
    """

    assert points_2d_screen.ndim == 2, "points_2d_screen must be (N, 2)"
    assert points_2d_screen.shape[1] == 2, "points_2d_screen must be (N, 2)"
    assert points_2d_screen.dtype == torch.float32, "points_2d_screen must be float32"
    if cameras_idx is not None:
        assert cameras_idx.ndim == 1, f"cameras_idx: {cameras_idx} must be 1D"
        if isinstance(cameras_idx, np.ndarray):
            assert cameras_idx.dtype == np.int32, "cameras_idx must be int32"
        if isinstance(cameras_idx, torch.Tensor):
            assert cameras_idx.dtype == torch.int32, "cameras_idx must be int32"
        assert (
            cameras_idx.shape[0] == points_2d_screen.shape[0]
        ), f"cameras_idx: {cameras_idx.shape[0]} must have the same length as points_2d_screen: {points_2d_screen.shape[0]}"
    if frames_idx is not None:
        assert frames_idx.ndim == 1, f"frames_idx must: {frames_idx} be 1D"
        if isinstance(frames_idx, np.ndarray):
            assert frames_idx.dtype == np.int32, "frames_idx must be int32"
        if isinstance(frames_idx, torch.Tensor):
            assert frames_idx.dtype == torch.int32, "frames_idx must be int32"
        assert (
            frames_idx.shape[0] == points_2d_screen.shape[0]
        ), f"frames_idx: {frames_idx.shape[0]} must have the same length as points_2d_screen: {points_2d_screen.shape[0]}"

    # convert to pixels
    pixels = points_2d_screen_to_pixels(points_2d_screen)

    return get_data_per_pixels(
        pixels=pixels,
        cameras_idx=cameras_idx,
        frames_idx=frames_idx,
        data_dict=data_dict,
    )
