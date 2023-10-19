import torch
import numpy as np


def decompose_projection_matrix(P):
    """
    Decompose a projection matrix into K, R, t such that P = K[R|t].
    Args:
        P (np.ndarray): 3x4 projection matrix.
    Returns:
        K (np.ndarray): 3x3 intrinsic matrix.
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
    """
    M = P[0:3, 0:3]
    Q = np.eye(3)[::-1]
    P_b = Q @ M @ M.T @ Q
    K_h = Q @ np.linalg.cholesky(P_b) @ Q
    K = K_h / K_h[2, 2]
    A = np.linalg.inv(K) @ M
    el = (1 / np.linalg.det(A)) ** (1 / 3)
    R = el * A
    t = el * np.linalg.inv(K) @ P[0:3, 3]
    return K, R, t


class Camera:
    """Camera class."""

    def __init__(
        self,
        imgs,
        intrinsics=None,
        pose=None,
        projection=None,
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

        if intrinsics is None and pose is None and projection is not None:
            K, R, t = decompose_projection_matrix(projection.numpy())
            Rt = np.eye(4)
            Rt[:3, :3] = R
            Rt[:3, 3] = t
            self.intrinsics = torch.from_numpy(K).float().to(device)
            self.pose = torch.from_numpy(Rt).float().to(device)
        elif intrinsics is not None and pose is not None:
            self.intrinsics = torch.from_numpy(intrinsics).float().to(device)
            self.pose = torch.from_numpy(pose).float().to(device)
        else:
            raise ValueError(
                "Either projection or intrinsics and pose must be provided"
            )

        self.intrinsics_inv = torch.inverse(self.intrinsics)
        self.imgs = torch.from_numpy(imgs).float().to(device)
        if masks is not None:
            self.masks = torch.from_numpy(masks).float().to(device)
        else:
            self.masks = None
        self.height = imgs.shape[1]
        self.width = imgs.shape[2]
        self.nr_pixels = self.height * self.width

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

    # def get_projection(self):
    #     """return camera projection matrix
    #     out: projection (3, 4)
    #     """
    #     # Perform the matrix multiplication
    #     projection = self.intrinsics @ self.get_pose()[:3, :4]
    #     return projection

    # def get_pose_inv(self):
    #     pose_inv = torch.inverse(self.get_pose())
    #     return pose_inv

    # def get_projection(self):
    #     pose = self.get_pose()
    #     projection = self.intrinsics @ pose
    #     return projection

    def concat_transform(self, transform):
        # apply transform
        self.transform = transform @ self.transform
        # # normalize
        # self.transform = self.transform / self.transform[3, 3]
        # # make sure rotation is still orthogonal
        # assert torch.linalg.det(self.transform[:3, :3]) - 1 < 1e-6

    # def get_rays(self):
    #     """
    #     Generate rays in world space for each pixel.

    #     Returns:
    #     rays_o (torch.tensor): centers of each ray. (H*W, 3)
    #     rays_d (torch.tensor): direction of each ray (a unit vector). (H*W, 3)
    #     """

    #     pose = self.get_pose()
    #     intrinsics_inv = torch.inverse(self.intrinsics)
    #     tx = torch.linspace(0, self.width - 1, 1)
    #     ty = torch.linspace(0, self.height - 1, 1)
    #     pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
    #     # W, H, 3
    #     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
    #     print(p.shape)
    #     # W, H, 3
    #     # p = torch.matmul(intrinsics_inv[None, :3, :3], p[:, :, :, None]).squeeze()
    #     # print(p.shape)
    #     # # W, H, 3
    #     # rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    #     # print(rays_d.shape)
    #     # # W, H, 3
    #     # rays_d = torch.matmul(pose[None, :3, :3], rays_d[:, :, :, None]).squeeze()
    #     # # W, H, 3
    #     # rays_o = pose[None, :3, 3].expand(rays_d.shape)

    #     # # (H, W), (H, W)
    #     # ii, jj = torch.meshgrid(
    #     #     torch.arange(0, self.width, 1),
    #     #     torch.arange(0, self.height, 1),
    #     #     indexing="xy",
    #     # )

    #     # # (H, W, 3)
    #     # local_rays_d = torch.stack(
    #     #     [
    #     #         (ii - self.width * 0.5) / self.intrinsics[0, 0],
    #     #         -(jj - self.height * 0.5) / self.intrinsics[1, 1],
    #     #         -torch.ones_like(ii),
    #     #     ],
    #     #     dim=-1,
    #     # )

    #     # rays_d = torch.sum(local_rays_d[..., None, :] * self.get_pose()[:3, :3], dim=-1)
    #     # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    #     # rays_o = self.get_pose()[:3, -1].expand(rays_d.shape)
    #     # rays_o = rays_o.view(-1, 3)
    #     # rays_d = rays_d.view(-1, 3)

    #     return rays_o, rays_d

    # def get_random_rays(self, nr_rays):
    #     """
    #     Generate N random rays in world space.

    #     Returns:
    #     rays_o (torch.tensor): centers of each ray. (N, 3)
    #     rays_d (torch.tensor): direction of each ray (a unit vector). (N, 3)
    #     """

    #     pixels_x = torch.randint(low=0, high=self.width, size=[nr_rays])
    #     pixels_y = torch.randint(low=0, high=self.height, size=[nr_rays])
    #     color = self.img[(pixels_y, pixels_x)]  # batch_size, 3
    #     p = torch.stack(
    #         [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
    #     ).float()  # batch_size, 3
    #     # p = torch.matmul(
    #     #     self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]
    #     # ).squeeze()  # batch_size, 3
    #     # rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
    #     # rays_v = torch.matmul(
    #     #     self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]
    #     # ).squeeze()  # batch_size, 3
    #     # rays_o = self.pose_all[img_idx, None, :3, 3].expand(
    #     #     rays_v.shape
    #     # )  # batch_size, 3

    #     return rays_o, rays_d, color
