import torch
import torch.nn.functional as F

from mvdatasets.utils.geometry import augment_vectors, inv_perspective_projection


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


def get_camera_rays_per_points_2d(c2w, intrinsics_inv, points_2d):
    """given a list of pixels, return rays origins and directions
    from a single camera

    args:
        c2w (torch.tensor): (4, 4)
        intrinsics_inv (torch.tensor): (3, 3)
        points_2d (torch.tensor, float): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
    """

    assert points_2d.shape[1] == 2, "pixels must be (N, 2)"

    # ray origin is just the camera center
    rays_o = c2w[:3, -1].unsqueeze(0).expand(points_2d.shape[0], -1)

    # pixels have height, width order
    points_3d_unprojected = inv_perspective_projection(intrinsics_inv, points_2d[:, [1, 0]])

    # normalize rays
    rays_d_camera = F.normalize(points_3d_unprojected, dim=-1)

    # rotate points to world space
    rays_d = (c2w[:3, :3] @ rays_d_camera.T).T

    return rays_o, rays_d


def get_camera_rays(camera, points_2d=None, device="cpu", jitter_pixels=False):
    """returns image rays origins and directions for pixels

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
        points_2d (torch.tensor, float): (N, 2) screen space sampling coordinates

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


def get_camera_frames_per_points_2d(points_2d, frame, mask=None):
    """given a list of pixels and a list of frames, return rgb and mask values at pixels

    args:
        points_2d (torch.tensor, float or int): (N, 2) with values in [0, height-1], [0, width-1]
        frame (torch.tensor): (height, width, 3)
        mask (torch.tensor, optional): (height, width, 1), default is None
        
    out:
        rgb (torch.tensor): (N, 3)
        mask_val (torch.tensor): (N, 1)
    """

    assert points_2d.shape[1] == 2, "pixels must be (N, 2)"
    
    if points_2d.dtype == torch.int32:
        # pixels have height, width order
        x, y = points_2d[:, 1], points_2d[:, 0]
        rgb = frame[y, x]
        mask_val = None
        if mask is not None:
            mask_val = mask[y, x]
        return rgb, mask_val
    
    # TODO: bilinear interpolation
    elif points_2d.dtype == torch.float32:
        points_2d = points_2d.int()
        x, y = points_2d[:, 1], points_2d[:, 0]
        rgb = frame[y, x]
        mask_val = None
        if mask is not None:
            mask_val = mask[y, x]
        return rgb, mask_val

    else:
        raise ValueError("points_2d must be int32 or float32")


def get_camera_frames(camera, points_2d=None, frame_idx=0, device="cpu", jitter_pixels=False):
    """returns camera images pixels values
    
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
        rgb (torch.tensor): (N, 3)
        mask_val (torch.tensor): (N, 1)
    
    """

    if points_2d is None:
        pixels = get_pixels(camera.height, camera.width, device=device)
        pixels = pixels.reshape(-1, 2)
        points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)

    frame = torch.from_numpy(
            camera.get_frame(frame_idx=frame_idx)
        ).float().to(device)
    mask = None
    if camera.has_masks:
        mask = torch.from_numpy(
                camera.get_mask(frame_idx=frame_idx)
            ).float().to(device)

    rgb, mask_val = get_camera_frames_per_points_2d(
        points_2d, frame, mask=mask
    )

    return rgb, mask_val, points_2d


def get_all_camera_rays_and_frames(camera, jitter_pixels=False, device="cpu"):
    """returns all camera rays and images pixels values
    
    jitter_pixels (bool, optional): whether to jitter pixels. Defaults to False.
    """

    pixels = get_pixels(camera.height, camera.width, device=device)
    pixels = pixels.reshape(-1, 2)
    points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)
    rays_o, rays_d, _ = get_camera_rays(
        camera, points_2d=points_2d, device=device
    )
    rgb, mask, _ = get_camera_frames(camera, points_2d=points_2d, device=device)
    return rays_o, rays_d, rgb, mask, points_2d


def get_random_camera_rays_and_frames(
    camera, nr_rays=512, frame_idx=0, jitter_pixels=False, device="cpu"
):
    """given a camera and a number of rays, return random
    rays and images pixels values
    
    jitter_pixels (bool, optional): whether to jitter pixels. Defaults to False.
    """

    pixels = get_random_pixels(camera.height, camera.width, nr_rays, device=device)
    points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)
    rays_o, rays_d, _ = get_camera_rays(
        camera, points_2d=points_2d, jitter_pixels=jitter_pixels, device=device
    )
    rgb, mask, _ = get_camera_frames(
        camera, points_2d=points_2d, frame_idx=frame_idx, device=device
    )

    return rays_o, rays_d, rgb, mask, points_2d

# TENSOR REEL -------------------------------------------------------------


def get_cameras_rays_per_pixel(c2w_all, intrinsics_inv_all, pixels, jitter_pixels=False):
    """given a list of c2w, intrinsics_inv and pixels, return rays origins and
    directions from multiple cameras

    args:
        c2w_all (torch.tensor): (N, 4, 4)
        intrinsics_inv_all (torch.tensor): (N, 3, 3)
        pixels (torch.tensor, int): (N, 2) with values in [0, width-1], [0, height-1]
        jitter_pixels (bool, optional): whether to jitter pixels. Defaults to False.

    out:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
    """

    # ray origin are the cameras centers
    rays_o = c2w_all[:, :3, -1]
    # print("rays_o", rays_o.shape, rays_o.device)

    points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)
    
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

    return rays_o, rays_d, points_2d


def get_cameras_frames_per_pixels(pixels, camera_idx, frame_idx, frames, masks=None):
    """given a list of pixels and a list of frames, return rgb and mask values at pixels

    args:
        pixels (torch.tensor, int): (N, 2) with values in [0, width-1], [0, height-1]

    out:
        rgb (torch.tensor): (N, 3)
        mask_val (torch.tensor): (N, 1)
    """

    assert pixels.shape[1] == 2, "pixels must be (N, 2)"

    # height = frames.shape[2]

    # get rgb and mask gt values at pixels
    rgb = frames[camera_idx, frame_idx, pixels[:, 0], pixels[:, 1]]
    # rgb = frames[camera_idx, frame_idx, (height - 1) - pixels[:, 0], pixels[:, 1]]
    mask_val = None
    if masks is not None:
        mask_val = masks[camera_idx, frame_idx, pixels[:, 0], pixels[:, 1]]
        # mask_val = masks[
        #    camera_idx, frame_idx, (height - 1) - pixels[:, 0], pixels[:, 1]
        # ]

    assert rgb.shape[1] == 3, "rgb must be (N, 3)"
    assert mask_val is None or mask_val.shape[1] == 1, "mask_val must be (N, 1)"

    return rgb, mask_val
