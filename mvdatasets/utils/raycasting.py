import torch
import numpy as np
import torch.nn.functional as F

from mvdatasets.utils.geometry import inv_perspective_projection


def get_pixels(height, width, device="cpu"):
    """returns all image pixels coords

    out:
        pixels (torch.tensor, int): (height, width, 2), values in [0, height-1], [0, width-1]
    """

    pixels_x, pixels_y = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    pixels = torch.stack([pixels_x, pixels_y], dim=-1).int()
    
    return pixels


def get_random_pixels(height, width, nr_pixels, device="cpu"):
    """given a number or pixels, return random pixels

    out:
        pixels (torch.tensor, int): (N, 2) with values in [0, width-1], [0, height-1]
    """
    # sample nr_pixels random pixels
    pixels = torch.rand(nr_pixels, 2, device=device)
    pixels[:, 0] *= height
    pixels[:, 1] *= width
    pixels = pixels.int()

    return pixels

# TODO: implement sampling from error map
# def get_pixels_sampled_from_error_map(error_map, height, width, nr_pixels, device="cpu"):
#     """given a number of pixels and an error map, sample pixels with error map as probability

#     Args:
#         error_map (torch.tensor): (height, width, 1) with values in [0, 1]
#         height (int): frame height
#         width (int): frame width
#         nr_pixels (int): number of pixels to sample
#         device (str, optional): Defaults to "cpu".
#     """

#     # TODO: use error map as probability to sample pixels
    
#     pixels = None
    
#     return pixels


def get_pixels_centers(pixels):
    """return the center of each pixel

    args:
        pixels (torch.tensor): (N, 2) list of pixels
    out:
        pixels_centers (torch.tensor): (N, 2) list of pixels centers
    """

    points_2d = pixels.float()
    points_2d = points_2d + 0.5  # pixels centers

    return points_2d


def jitter_points(points, std=0.12):
    """apply noise to points

    Args:
        points (torch.tensor): (N, 2) list of pixels centers (in screen space)
    Out:
        jittered_pixels (torch.tensor): (N, 2) list of pixels
    """

    # if pixels are int, convert to float:
    jittered_points = points
    # sample offsets from gaussian distribution
    offsets = torch.normal(
        mean=0.0, std=std, size=jittered_points.shape, device=points.device
    )
    # clamp offsets to [-0.5, 0.5]
    offsets = torch.clamp(offsets, -0.5, 0.5)
    # uniformlu sampled offsets
    # offsets = torch.rand_like(jittered_points, device=jittered_points.device) - 0.5
    jittered_points += offsets

    return jittered_points


def get_points_2d_from_pixels(pixels, jitter_pixels):
    """convert pixels to 2d points on the image plane"""
    assert pixels.dtype == torch.int32, "pixels must be int32"
    # get pixels as 3d points on a plane at z=-1 (in camera space)
    points_2d = get_pixels_centers(pixels)
    if jitter_pixels:
        points_2d = jitter_points(points_2d)
    return points_2d


# SINGLE CAMERA ----------------------------------------------------------


def get_camera_rays_per_points_2d(c2w, intrinsics_inv, points_2d_screen):
    """given a list of pixels, return rays origins and directions
    from a single camera

    args:
        c2w (torch.tensor): (4, 4)
        intrinsics_inv (torch.tensor): (3, 3)
        points_2d_screen (torch.tensor, float): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
    """

    assert points_2d_screen.shape[1] == 2, "points_2d_screen must be (N, 2)"

    # ray origin is just the camera center
    rays_o = c2w[:3, -1].unsqueeze(0).expand(points_2d_screen.shape[0], -1)

    # pixels have height, width order
    points_3d_camera = inv_perspective_projection(
        intrinsics_inv,
        points_2d_screen[:, [1, 0]]  # pixels have h, w order but we need x, y
    )
    # points_3d_unprojected have all z=1
    
    # rotate points to world space
    points_3d_world = (c2w[:3, :3] @ points_3d_camera.T).T
    
    rays_d = F.normalize(points_3d_world, dim=-1)

    # print("rays_d", rays_d)

    return rays_o, rays_d


def get_camera_rays(camera, points_2d=None, device="cpu", jitter_pixels=False):
    """returns image rays origins and directions
    for 2d points on the image plane.
    If points are not provided, they are sampled 
    from the image plane for every pixel.

    args:
        camera (Camera): camera object
        points_2d (torch.tensor, float or int, optional): (N, 2)
                                            Values in [0, height-1], [0, width-1].
                                            Default is None.
        device (str, optional): device to store tensors. Defaults to "cpu".
        jitter_pixels (bool, optional): Whether to jitter pixels.
                                        Only used if points_2d is None.
                                        Defaults to False.
    out:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
        points_2d (torch.tensor, float): (N, 2) screen space
                                        sampling coordinates

    """

    if points_2d is None:
        pixels = get_pixels(camera.height, camera.width, device=device)
        pixels = pixels.reshape(-1, 2)
        points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)

    c2w = torch.from_numpy(camera.get_pose()).float().to(device)
    intrinsics_inv = torch.from_numpy(
        camera.get_intrinsics_inv()
    ).float().to(device)
    
    rays_o, rays_d = get_camera_rays_per_points_2d(
        c2w, intrinsics_inv, points_2d
    )

    return rays_o, rays_d, points_2d


def get_camera_frames_per_points_2d(points_2d, rgb=None, mask=None):
    """given a list of pixels and a list of frames, return rgb and mask values at pixels

    args:
        points_2d (torch.tensor, float or int): (N, 2) with values in [0, height-1], [0, width-1]
        rgb (torch.tensor): (height, width, 3)
        mask (torch.tensor, optional): (height, width, 1), default is None
        
    out:
        vals (dict):
            rgb_val (optional, torch.tensor): (N, 3)
            mask_val (optional, torch.tensor): (N, 1)
    """

    assert points_2d.shape[1] == 2, "points_2d must be (N, 2)"
    
    pixels = points_2d.int()  # floor
    x, y = pixels[:, 1], pixels[:, 0]
    
    # prepare output
    vals = {}
    
    # rgb
    if rgb is not None:
        rgb_val = rgb[y, x]
        vals["rgb"] = rgb_val
        
    # mask
    mask_val = None
    if mask is not None:
        mask_val = mask[y, x]
        vals["mask"] = mask_val
        
    # TODO: get other frame modalities
    
    # TODO: bilinear interpolation
    # if points_2d.dtype == torch.float32:
        # ...
    
    assert rgb_val is None or rgb_val.shape[1] == 3, "rgb must be (N, 3)"
    assert mask_val is None or mask_val.shape[1] == 1, "mask_val must be (N, 1)"
        
    return vals


def get_camera_frames(camera, points_2d=None, frame_idx=0, device="cpu", jitter_pixels=False):
    """returns camera images pixels values
    
    args:
        camera (Camera): camera object
        points_2d (torch.tensor, float or int, optional): (N, 2)
                                            Values in [0, height-1], [0, width-1].
                                            Default is None.
        frame_idx (int, optional): frame index. Defaults to 0.
        device (str, optional): device to store tensors. Defaults to "cpu".
        jitter_pixels (bool, optional): Whether to jitter pixels.
                                        Only used if points_2d is None.
                                        Defaults to False.
                                        
    out:
        vals (dict):
            rgb_val (optional, torch.tensor): (N, 3)
            mask_val (optional, torch.tensor): (N, 1)
        points_2d (torch.tensor, float): (N, 2)
                                        Values in [0, height-1], [0, width-1].
    """

    if points_2d is None:
        pixels = get_pixels(camera.height, camera.width, device=device)
        pixels = pixels.reshape(-1, 2)
        points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)

    # rgb
    rgb = None
    if camera.has_rgbs():
        rgb = torch.from_numpy(
                camera.get_rgb(frame_idx=frame_idx)
            ).float().to(device)
    
    # mask
    mask = None
    if camera.has_masks():
        mask = torch.from_numpy(
                camera.get_mask(frame_idx=frame_idx)
            ).float().to(device)
        
    # TODO: get other frames

    vals = get_camera_frames_per_points_2d(
        points_2d, rgb=rgb, mask=mask
    )

    return vals, points_2d


# TODO: deprecated
# def get_all_camera_rays_and_frames(camera, jitter_pixels=False, device="cpu"):
#     """returns all camera rays and images pixels values
    
#     jitter_pixels (bool, optional): whether to jitter pixels. Defaults to False.
#     """

#     pixels = get_pixels(camera.height, camera.width, device=device)
#     pixels = pixels.reshape(-1, 2)
#     points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)
#     rays_o, rays_d, _ = get_camera_rays(
#         camera, points_2d=points_2d, device=device
#     )
#     rgb, mask, _ = get_camera_frames(camera, points_2d=points_2d, device=device)
#     return rays_o, rays_d, rgb, mask, points_2d


def get_random_camera_rays_and_frames(
    camera, nr_rays=512, frame_idx=0, jitter_pixels=False, device="cpu"
):
    """given a camera and a number of rays, return random
    rays and images pixels values
    
    jitter_pixels (bool, optional): whether to jitter pixels. Defaults to False.
    """

    pixels = get_random_pixels(
        camera.height, camera.width, nr_rays, device=device
    )
    points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)
    
    rays_o, rays_d, points_2d = get_camera_rays(
        camera, points_2d=points_2d, device=device
    )
    rgb, mask, _ = get_camera_frames(
        camera, points_2d=points_2d, frame_idx=frame_idx, device=device
    )

    return rays_o, rays_d, rgb, mask, points_2d

# TENSOR REEL -------------------------------------------------------------


def get_cameras_rays_per_points_2d(c2w_all, intrinsics_inv_all, points_2d):
    """given a list of c2w, intrinsics_inv and points_2d, return rays origins and
    directions from multiple cameras

    args:
        c2w_all (torch.tensor): (N, 4, 4)
        intrinsics_inv_all (torch.tensor): (N, 3, 3)
        points_2d (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
    """
    
    assert points_2d.shape[1] == 2, "points_2d must be (N, 2)"

    # ray origin are the cameras centers
    rays_o = c2w_all[:, :3, -1]
    # print("rays_o", rays_o.shape, rays_o.device)
    
    # pixels_2d_ = points_2d[:, [1, 0]]  # pixels have width, height order
    pixels_3d = torch.cat(
        [points_2d, torch.ones_like(points_2d[:, 0]).unsqueeze(-1)], dim=-1
    )

    # from screen to camera coords
    points_3d_camera = intrinsics_inv_all @ pixels_3d.unsqueeze(-1)
    points_3d_camera = points_3d_camera.reshape(-1, 3)

    # normalize rays
    rays_d_camera = F.normalize(points_3d_camera, dim=-1)

    # rotate points to world space
    rays_d = c2w_all[:, :3, :3] @ rays_d_camera.unsqueeze(-1)
    rays_d = rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_cameras_frames_per_points_2d(points_2d, camera_idx, frame_idx, rgbs=None, masks=None):
    """given a list of 2d points on the image plane and a list of rgbs,
    return rgb and mask values at pixels

    args:
        points_2d (torch.tensor, float or int): (N, 2)
                                            Values in [0, height-1], [0, width-1].
        camera_idx (int): camera index
        frame_idx (int): frame index.
        rgbs (optional, torch.tensor): (N, T, H, W, 3) in [0, 1] or None
        masks (optional, torch.tensor): (N, T, H, W, 1) in [0, 1] or None
        

    out:
        vals (dict):
            rgb_val (optional, torch.tensor): (N, 3)
            mask_val (optional, torch.tensor): (N, 1)
    """

    assert points_2d.shape[1] == 2, "points_2d must be (N, 2)"

    pixels = points_2d.int()  # floor
    x, y = pixels[:, 1], pixels[:, 0]

    # prepare output
    vals = {}
    
    # rgb
    rgb_val = None
    if rgbs is not None:
        rgb_val = rgbs[camera_idx, frame_idx, y, x]
        vals["rgb"] = rgb_val
    
    # mask
    mask_val = None
    if masks is not None:
        mask_val = masks[camera_idx, frame_idx, y, x]
        vals["mask"] = mask_val
        
    # TODO: get other frame modalities
    
    # TODO: bilinear interpolation
    # if points_2d.dtype == torch.float32:
        # ...

    assert rgb_val is None or rgb_val.shape[1] == 3, "rgb must be (N, 3)"
    assert mask_val is None or mask_val.shape[1] == 1, "mask_val must be (N, 1)"

    return vals
