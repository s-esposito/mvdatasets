import numpy as np
import cv2 as cv

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
        self, imgs, intrinsics, pose, masks=None, global_transform=None, local_transform=None, camera_idx=0
    ):
        """Create a camera object, all parameters are np.ndarrays.

        Args:
            imgs (np.array): (T, H, W, 3) with values in [0, 1]
            intrinsics (np.array): (3, 3)
            pose (np.array): (4, 4)
            masks (np.array): (T, H, W, 1) with values in [0, 1]
        """

        # assert shapes are correct
        assert imgs.ndim == 4 and imgs.shape[-1] == 3
        assert intrinsics.shape == (3, 3)
        assert pose.shape == (4, 4)
        if masks is not None:
            assert masks.ndim == 4 and masks.shape[-1] == 1
        
        self.camera_idx = camera_idx
        self.intrinsics = intrinsics
        self.pose = pose
        self.intrinsics_inv = np.linalg.inv(intrinsics)
        self.imgs = imgs
        if masks is not None:
            self.has_masks = True
            self.masks = masks
        else:
            self.has_masks = False
            self.masks = None

        self.height = imgs.shape[1]
        self.width = imgs.shape[2]
        # self.nr_pixels = self.height * self.width
        # self.nr_frames = imgs.shape[0]

        if global_transform is not None:
            self.global_transform = global_transform
        else:
            self.global_transform = np.eye(4)
            
        if local_transform is not None:
            self.local_transform = local_transform
        else:
            self.local_transform = np.eye(4)

    def get_intrinsics(self):
        """return camera intrinsics"""
        return self.intrinsics

    def get_intrinsics_inv(self):
        """return inverse of camera intrinsics"""
        return self.intrinsics_inv

    def get_frames(self):
        """return all camera frames"""
        return self.imgs

    def get_frame(self, frame_idx=0):
        """returns image at frame_idx"""
        img = self.imgs[frame_idx]
        return img

    def get_masks(self):
        """return, if exists, all camera masks, else None"""
        return self.masks

    def get_mask(self, frame_idx=0):
        """return, if exists, a mask at frame_idx, else None"""
        mask = None
        if len(self.masks) > frame_idx:
            mask = self.masks[frame_idx]
        return mask

    def get_pose(self):
        """returns camera pose in world space"""
        pose = self.global_transform @ self.pose @ self.local_transform
        return pose

    def concat_global_transform(self, global_transform):
        # apply global_transform
        self.global_transform = global_transform @ self.global_transform

    def subsample(self, scale):
        """subsample camera frames and masks by scale (inplace operation)"""

        # update dimentions
        # new_height = self.height // scale
        # new_width = self.width // scale

        # subsample frames
        new_imgs = []
        for img in self.imgs:
            new_imgs.append(
                cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            )
        self.imgs = np.stack(new_imgs)

        # subsample masks
        if self.has_masks:
            new_masks = []
            for mask in self.masks:
                new_masks.append(
                    cv.resize(
                        mask, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA
                    )[..., None]
                )
            self.masks = np.stack(new_masks)

        self.height = self.imgs.shape[1]
        self.width = self.imgs.shape[2]

        # modify intrinsics
        self.intrinsics[0, 0] *= scale
        self.intrinsics[1, 1] *= scale
        self.intrinsics[0, 2] *= scale
        self.intrinsics[1, 2] *= scale
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

    def __str__(self):
        """print camera information"""
        string = ""
        string += "intrinsics:\n"
        string += str(self.intrinsics) + "\n"
        string += "pose:\n"
        string += str(self.pose) + "\n"
        string += "global_transform:\n"
        string += str(self.global_transform) + "\n"
        string += "local_transform:\n"
        string += str(self.local_transform) + "\n"
        string += "imgs:\n"
        string += str(self.imgs.shape) + "\n"
        if self.has_masks:
            string += "masks:\n"
            string += str(self.masks.shape) + "\n"

        return string
