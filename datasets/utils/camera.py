import torch
import numpy as np
import cv2 as cv


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def decompose_projection_matrix(c2w):
    """
    Decompose a projection matrix (c2w) into K, R, t such that P = K[R|t].
    Args:
        c2w (np.ndarray): 3x4 projection matrix.
    Returns:
        intrinsics (np.ndarray): 3x3 intrinsic matrix.
        pose (np.ndarray): 4x4 extrinsic matrix.
    """

    res = cv.decomposeProjectionMatrix(c2w)
    camera_matrix = res[0]
    rot = res[1]
    trasl = res[2]

    intrinsics = camera_matrix / camera_matrix[2, 2]  # normalize K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot.transpose()
    pose[:3, 3] = (trasl[:3] / trasl[3])[:, 0]

    return intrinsics, pose


class Camera:
    """Camera class."""

    def __init__(self, c2w, img, mask=None, transform=None):
        """Create a camera object

        Args:
            c2w (torch.tensor): projection matrix (3, 4)
            img (torch.tensor): (H, W, 3)
            mask (torch.tensor): (H, W, 1)
        """

        self.c2w = c2w
        # K: (3, 3), [R|t]: (4, 4)
        self.intrinsics, self.pose = decompose_projection_matrix(self.c2w.cpu().numpy())
        self.intrinsics = torch.from_numpy(self.intrinsics)
        self.pose = torch.from_numpy(self.pose)
        self.img = img  # torch.tensor of shape (H, W, 3)
        self.mask = mask  # torch.tensor of shape (H, W, 1)
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.transform = transform if transform is not None else torch.eye(4)

    def get_pose(self):
        return self.transform @ self.pose

    def concat_transform(self, transform):
        self.transform = transform @ self.transform

    # def get_rays(self):
    #     """
    #     Generate rays in world space for each pixel.

    #     Returns:
    #     rays_o (torch.tensor): centers of each ray. (H*W, 3)
    #     rays_d (torch.tensor): direction of each ray (a unit vector). (H*W, 3)
    #     """

    #     # (H, W), (H, W)
    #     ii, jj = torch.meshgrid(
    #         torch.arange(0, self.width, self.width),
    #         torch.arange(0, self.height, self.height),
    #         indexing="xy",
    #     )

    #     # (H, W, 3)
    #     local_rays_d = torch.stack(
    #         [
    #             (ii - self.width * 0.5) / self.K[0, 0],
    #             -(jj - self.height * 0.5) / self.K[1, 1],
    #             -torch.ones_like(ii),
    #         ],
    #         dim=-1,
    #     )

    #     rays_d = torch.sum(local_rays_d[..., None, :] * self.c2w[:3, :3], dim=-1)
    #     rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    #     rays_o = self.c2w[:3, -1].expand(rays_d.shape)

    #     # (N_rays, 3)
    #     rays_o = rays_o.view(-1, 3)
    #     rays_d = rays_d.view(-1, 3)

    #     return rays_o, rays_d

    # def get_random_rays(self, nr_rays):
    #     """
    #     Generate N random rays in world space.

    #     Returns:
    #     rays_o (torch.tensor): centers of each ray. (N, 3)
    #     rays_d (torch.tensor): direction of each ray (a unit vector). (N, 3)
    #     """
    #     rays_o, rays_d = self.get_rays()
    #     idxs = np.random.choice(rays_o.shape[0], nr_rays, replace=False)
    #     rays_o = rays_o[idxs]
    #     rays_d = rays_d[idxs]
    #     return rays_o, rays_d
