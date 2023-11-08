import torch
import torch.nn.functional as F


def get_pixels(height, width, device="cpu"):
    """returns all image pixels coords"""

    pixels_x, pixels_y = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    pixels = torch.stack([pixels_x, pixels_y], dim=-1).reshape(-1, 2).int()

    return pixels


def get_random_pixels(height, width, nr_pixels, device="cpu"):
    """given a number or pixels, return random pixels

    out:
        pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]
    """
    # sample nr_rays random pixels
    pixels = torch.rand(nr_pixels, 2, device=device)
    pixels[:, 0] *= height
    pixels[:, 1] *= width
    pixels = pixels.int()

    return pixels


def get_camera_rays_per_pixels(c2w, intrinsics_inv, pixels):
    """given a list of pixels, return rays origins and directions
    from a single camera

    args:
        c2w (torch.tensor): (4, 4)
        intrinsics_inv (torch.tensor): (3, 3)
        pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
    """
    # ray origin is just the camera center
    # c2w = self.get_pose()
    rays_o = c2w[:3, -1].unsqueeze(0).expand(pixels.shape[0], -1)

    # get pixels as 3d points on a plane at z=-1 (in camera space)
    pixels = pixels.float()
    points_3d_camera = torch.stack(
        [
            pixels[:, 1] * intrinsics_inv[0, 0],
            pixels[:, 0] * intrinsics_inv[1, 1],
            -1 * torch.ones_like(pixels[:, 0]),
        ],
        dim=-1,
    )

    # normalize rays
    rays_d_camera = F.normalize(points_3d_camera, dim=-1)

    # rotate points to world space
    rays_d = (c2w[:3, :3] @ rays_d_camera.T).T

    return rays_o, rays_d


def get_camera_rays(camera, pixels=None, device="cpu"):
    """returns all image rays origins and directions from a single camera

    args:
        camera (Camera): camera object
        pixels (torch.tensor, int, optional): (N, 2) with values in
                                            [0, height-1], [0, width-1], default is None
        device (str, optional): device to store tensors. Defaults to "cpu".

    out:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)

    """

    if pixels is None:
        pixels = get_pixels(camera.height, camera.width, device=device)

    c2w = torch.from_numpy(camera.get_pose()).float().to(device)
    intrinsics_inv = torch.from_numpy(camera.get_intrinsics_inv()).float().to(device)

    rays_o, rays_d = get_camera_rays_per_pixels(c2w, intrinsics_inv, pixels)

    return rays_o, rays_d


def get_cameras_rays_per_pixel(c2w_all, intrinsics_inv_all, pixels):
    """given a list of c2w, intrinsics_inv and pixels, return rays origins and
    directions from multiple cameras

    args:
        c2w_all (torch.tensor): (N, 4, 4)
        intrinsics_inv_all (torch.tensor): (N, 3, 3)
        pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
    """

    # ray origin are the cameras centers
    rays_o = c2w_all[:, :3, -1]
    # print("rays_o", rays_o.shape, rays_o.device)

    # get pixels as 3d points on a plane at z=-1 (in camera space)
    pixels = pixels.float()
    points_3d_camera = torch.stack(
        [
            pixels[:, 1] * intrinsics_inv_all[:, 0, 0],
            pixels[:, 0] * intrinsics_inv_all[:, 1, 1],
            -1 * torch.ones_like(pixels[:, 0]),
        ],
        dim=-1,
    )

    # normalize rays
    rays_d_camera = F.normalize(points_3d_camera, dim=-1)
    # print("rays_d_camera", rays_d_camera.shape, rays_d_camera.device)

    # rotate points to world space
    rays_d = c2w_all[:, :3, :3] @ rays_d_camera.unsqueeze(-1)
    rays_d = rays_d.reshape(-1, 3)
    # print("rays_d", rays_d.shape, rays_d.device)

    return rays_o, rays_d


def get_camera_frames_per_pixels(pixels, frame, mask=None):
    """given a list of pixels and a list of frames, return rgb and mask values at pixels

    args:
        pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rgb (torch.tensor): (N, 3)
        gray (torch.tensor): (N, 1)
    """

    # rgb = frame[pixels[:, 0], pixels[:, 1]]
    # mask_val = None
    # if mask is not None:
    #     mask_val = mask[pixels[:, 0], pixels[:, 1]]
    # mask_val = mask_val.unsqueeze(-1)

    # camera image plane is flipped vertically
    height = frame.shape[0]
    rgb = frame[(height - 1) - pixels[:, 0], pixels[:, 1]]
    mask_val = None
    if mask is not None:
        mask_val = mask[(height - 1) - pixels[:, 0], pixels[:, 1]]
    mask_val.unsqueeze(-1)

    return rgb, mask_val


def get_camera_frames(camera, pixels=None, frame_idx=0, device="cpu"):
    """returns all camera images pixels values"""

    if pixels is None:
        pixels = get_pixels(camera.height, camera.width, device=device)

    frame = torch.from_numpy(camera.get_frame(frame_idx=frame_idx)).float().to(device)
    mask = None
    if camera.has_masks:
        mask = torch.from_numpy(camera.get_mask(frame_idx=frame_idx)).float().to(device)

    rgb, mask_val = get_camera_frames_per_pixels(pixels, frame, mask=mask)

    return rgb, mask_val


def get_camera_rays_and_frames(camera, device="cpu"):
    """returns all camera rays and images pixels values"""
    rays_o, rays_d = get_camera_rays(camera, device=device)
    rgb, mask = get_camera_frames(camera, device=device)
    return rays_o, rays_d, rgb, mask


def get_camera_random_rays_and_frames(camera, nr_rays=512, frame_idx=0, device="cpu"):
    """given a camera and a number of rays,
    return random rays and images pixels values"""

    pixels = get_random_pixels(camera.height, camera.width, nr_rays, device=device)

    rays_o, rays_d = get_camera_rays(camera, pixels=pixels, device=device)
    rgb, mask = get_camera_frames(camera, pixels=pixels, frame_idx=0, device=device)

    return rays_o, rays_d, rgb, mask


def get_cameras_frames_per_pixels(pixels, camera_idx, frame_idx, frames, masks=None):
    """given a list of pixels and a list of frames, return rgb and mask values at pixels

    args:
        pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rgb (torch.tensor): (N, 3)
        gray (torch.tensor): (N, 1)
    """

    height = frames.shape[2]

    # get rgb and mask gt values at pixels
    # rgb = frames[camera_idx, frame_idx, pixels[:, 0], pixels[:, 1]]
    rgb = frames[camera_idx, frame_idx, (height - 1) - pixels[:, 0], pixels[:, 1]]
    mask_val = None
    if masks is not None:
        # mask_val = masks[camera_idx, frame_idx, pixels[:, 0], pixels[:, 1]]
        mask_val = masks[
            camera_idx, frame_idx, (height - 1) - pixels[:, 0], pixels[:, 1]
        ]

    return rgb, mask_val
