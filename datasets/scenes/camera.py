import torch
import torch.nn.functional as F
import numpy as np


# def decompose_projection_matrix(P):
#     """
#     Decompose a projection matrix into K, R, t such that P = K[R|t].
#     Args:
#         P (np.ndarray): 3x4 projection matrix.
#     Returns:
#         K (np.ndarray): 3x3 intrinsic matrix.
#         R (np.ndarray): 3x3 rotation matrix.
#         t (np.ndarray): 3x1 translation vector.
#     """
#     M = P[0:3, 0:3]
#     Q = np.eye(3)[::-1]
#     P_b = Q @ M @ M.T @ Q
#     K_h = Q @ np.linalg.cholesky(P_b) @ Q
#     K = K_h / K_h[2, 2]
#     A = np.linalg.inv(K) @ M
#     el = (1 / np.linalg.det(A)) ** (1 / 3)
#     R = el * A
#     t = el * np.linalg.inv(K) @ P[0:3, 3]
#     return K, R, t


class Camera:
    """Camera class."""

    def __init__(
        self,
        imgs,
        intrinsics,
        pose,
        # projection=None,
        masks=None,
        transform=None,
        device="cuda:0",
    ):
        """Create a camera object, all parameters are torch tensors.

        Args:
            imgs (np.array): (T, H, W, 3) with values in [0, 1]
            intrinsics (np.array): (3, 3)
            pose (np.array): (4, 4)
            projection (np.array): projection matrix (3, 4)
            masks (np.array): (T, H, W, 1) with values in [0, 1]
        """

        # if intrinsics is None and pose is None and projection is not None:
        #     K, R, t = decompose_projection_matrix(projection.numpy())
        #     Rt = np.eye(4)
        #     Rt[:3, :3] = R
        #     Rt[:3, 3] = t
        #     self.intrinsics = torch.from_numpy(K).float().to(device)
        #     self.pose = torch.from_numpy(Rt).float().to(device)
        # elif intrinsics is not None and pose is not None:
        self.intrinsics = torch.from_numpy(intrinsics).float().to(device)
        self.pose = torch.from_numpy(pose).float().to(device)
        # else:
        #     raise ValueError(
        #         "Either projection or intrinsics and pose must be provided"
        #     )

        self.intrinsics_inv = torch.inverse(self.intrinsics)
        self.imgs = torch.from_numpy(imgs).float().to(device)
        if masks is not None:
            self.masks = torch.from_numpy(masks).float().to(device)
        else:
            self.masks = None
        self.height = imgs.shape[1]
        self.width = imgs.shape[2]
        self.nr_pixels = self.height * self.width
        self.nr_frames = imgs.shape[0]

        if transform is not None:
            self.transform = torch.from_numpy(transform).float().to(device)
        else:
            self.transform = torch.from_numpy(np.eye(4)).float().to(device)

        # # print each tensor device
        # print("transform", self.transform.device)
        # print("pose", self.pose.device)
        # print("intrinsics", self.intrinsics.device)
        # print("intrinsics_inv", self.intrinsics_inv.device)
        # print("imgs", self.imgs.device)
        # if self.masks is not None:
        #     print("masks", self.masks.device)

    def get_intrinsics(self):
        """return camera intrinsics"""
        return self.intrinsics

    def get_frame(self, timestamp=0):
        """returns image at timestamp"""
        return self.imgs[timestamp]

    def get_mask(self, timestamp=0):
        """return, if exists, a mask at timestamp, else None"""
        mask = None
        if len(self.masks) > timestamp:
            mask = self.masks[timestamp]
        return mask

    def get_pose(self):
        """returns camera pose in world space"""
        pose = self.transform @ self.pose
        return pose

    def concat_transform(self, transform):
        # apply transform
        self.transform = transform @ self.transform

    def get_rays_per_pixels(self, pixels):
        """given a list of pixels, return rays origins and directions

        args:
            pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

        out:
            rays_o (torch.tensor): (N, 3)
            rays_d (torch.tensor): (N, 3)
        """
        # ray origin is just the camera center
        c2w = self.get_pose()
        rays_o = c2w[:3, -1].unsqueeze(0).expand(pixels.shape[0], -1)

        # get pixels as 3d points on a plane at z=-1 (in camera space)
        pixels = pixels.float()
        points_3d_camera = torch.stack(
            [
                pixels[:, 1] * self.intrinsics_inv[0, 0],
                pixels[:, 0] * self.intrinsics_inv[1, 1],
                -1 * torch.ones_like(pixels[:, 0]),
            ],
            dim=-1,
        )

        # normalize rays
        rays_d_camera = F.normalize(points_3d_camera, dim=-1)

        # rotate points to world space
        rays_d = (c2w[:3, :3] @ rays_d_camera.T).T

        return rays_o, rays_d

    def get_random_pixels(self, nr_pixels):
        """given a number or pixels, return random pixels

        out:
            pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]
        """
        # sample nr_rays random pixels
        pixels = torch.rand(nr_pixels, 2)
        pixels[:, 0] *= self.height
        pixels[:, 1] *= self.width
        pixels = pixels.int()

        return pixels

    def get_random_rays(self, nr_rays):
        """given a number or rays, return rays origins and random directions

        args:
            nr_rays (int): number or random rays to sample

        out:
            rays_o (torch.tensor): (N, 3)
            rays_d (torch.tensor): (N, 3)
        """

        pixels = self.get_random_pixels(nr_rays)
        return self.get_rays_per_pixels(pixels)

    def get_frame_per_pixels(self, pixels, timestamp=0):
        """given a list of pixels, return the corresponding frame values

        args:
            pixels (torch.tensor, int): (N, 2) with values in [0, height-1], [0, width-1]

        out:
            rgb (torch.tensor): (N, 3)
            alpha (torch.tensor): (N, 1)
        """
        img = self.get_frame(timestamp)
        mask = self.get_mask(timestamp)

        # camera image plane is flipped vertically
        rgb = img[(self.height - 1) - pixels[:, 0], pixels[:, 1]]
        alpha = mask[(self.height - 1) - pixels[:, 0], pixels[:, 1]]

        return rgb, alpha
