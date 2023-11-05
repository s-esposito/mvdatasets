import torch
import torch.nn.functional as F


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


def get_frame_per_pixels(pixels, frames, mask=None):
    """given a list of pixels and a list of frames, return rgb and mask values at pixels

    args:
        pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rgb (torch.tensor): (N, 3)
        gray (torch.tensor): (N, 1)
    """

    height = frames.shape[0]
    # width = frame.shape[1]

    # camera image plane is flipped vertically
    rgb = frames[(height - 1) - pixels[:, 0], pixels[:, 1]]
    gray = None
    if mask is not None:
        gray = mask[(height - 1) - pixels[:, 0], pixels[:, 1]]

    return rgb, gray


def get_cameras_frames_per_pixels(camera_idx, frame_idx, pixels, frames, mask=None):
    """TODO

    args:
        pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

    out:
        rgb (torch.tensor): (N, 3)
        gray (torch.tensor): (N, 1)
    """

    height = frames.shape[2]
    # width = frames.shape[3]

    # camera image plane is flipped vertically
    rgb = frames[camera_idx, frame_idx, (height - 1) - pixels[:, 0], pixels[:, 1]]
    gray = None
    if mask is not None:
        gray = mask[camera_idx, frame_idx, (height - 1) - pixels[:, 0], pixels[:, 1]]

    return rgb, gray
