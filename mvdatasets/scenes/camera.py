import numpy as np
import cv2 as cv

# from mvdatasets.utils.raycasting import (
#     get_camera_rays_per_pixels,
#     get_random_pixels,
#     get_frame_per_pixels,
# )

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

    def __init__(self, imgs, intrinsics, pose, masks=None, transform=None):
        """Create a camera object, all parameters are np.ndarrays.

        Args:
            imgs (np.array): (T, H, W, 3) with values in [0, 1]
            intrinsics (np.array): (3, 3)
            pose (np.array): (4, 4)
            masks (np.array): (T, H, W, 1) with values in [0, 1]
        """

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

        if transform is not None:
            self.transform = transform
        else:
            self.transform = np.eye(4)

    def get_intrinsics(self):
        """return camera intrinsics"""
        return self.intrinsics

    def get_intrinsics_inv(self):
        """return inverse of camera intrinsics"""
        return self.intrinsics_inv

    def get_frames(self):
        """return all camera frames"""
        return self.imgs

    def get_frame(self, timestamp=0, subsampling_factor=1.0):
        """returns image at timestamp"""
        img = self.imgs[timestamp]
        return img

    def get_masks(self):
        """return, if exists, all camera masks, else None"""
        return self.masks

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
                    )
                )
            self.masks = np.stack(new_masks)

        self.height = self.imgs.shape[1]
        self.width = self.imgs.shape[2]

        # modify intrinsics
        self.intrinsics = self.intrinsics * scale
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

    def __str__(self):
        """print camera information"""
        string = ""
        string += "intrinsics:\n"
        string += str(self.intrinsics) + "\n"
        string += "pose:\n"
        string += str(self.pose) + "\n"
        string += "transform:\n"
        string += str(self.transform) + "\n"
        string += "imgs:\n"
        string += str(self.imgs.shape) + "\n"
        if self.has_masks:
            string += "masks:\n"
            string += str(self.masks.shape) + "\n"

        return string

    # def get_random_rays(self, nr_rays):
    #     """given a number or rays, return rays origins and random directions

    #     args:
    #         nr_rays (int): number or random rays to sample

    #     out:
    #         rays_o (np.ndarray): (N, 3)
    #         rays_d (np.ndarray): (N, 3)
    #     """

    #     pixels = get_random_pixels(self.height, self.width, nr_rays)
    #     return get_camera_rays_per_pixels(self.get_pose(), self.intrinsics_inv, pixels)

    # def get_random_pixels(self, nr_pixels):
    #     """given a number or pixels, return random pixels

    #     out:
    #         pixels (np.ndarray, int): (N, 2) with values in [0, height-1], [0, width-1]
    #     """
    #     return get_random_pixels(self.height, self.width, nr_pixels)

    # def get_rays_per_pixels(self, pixels):
    #     """given a list of pixels, return rays origins and directions

    #     args:
    #         pixels (np.ndarray, int): (N, 2) with values in [0, height-1], [0, width-1]

    #     out:
    #         rays_o (np.ndarray): (N, 3)
    #         rays_d (np.ndarray): (N, 3)
    #     """
    #     return get_camera_rays_per_pixels(self.get_pose(), self.intrinsics_inv, pixels)

    # def get_frame_per_pixels(self, pixels, timestamp=0):
    #     """given a list of pixels, return color and mask values

    #     args:
    #         pixels (np.ndarray, int): (N, 2) with values in [0, height-1], [0, width-1]

    #     out:
    #         rgb (np.ndarray): (N, 3)
    #         mask (np.ndarray): (N, 1)
    #     """
    #     return get_frame_per_pixels(
    #         self.get_frame(timestamp), self.get_mask(timestamp), pixels
    #     )
