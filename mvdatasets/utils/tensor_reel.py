import torch
import numpy as np
from tqdm import tqdm

from mvdatasets.utils.images import image_uint8_to_float32
from mvdatasets.utils.raycasting import (
    get_pixels,
    get_random_pixels,
    get_rays_per_points_2d_screen,
    get_data_per_points_2d_screen,
    get_points_2d_screen_from_pixels,
    get_random_pixels_from_error_map,
)
from mvdatasets import Camera
from mvdatasets.utils.printing import print_error, print_info


class TensorReel:
    def __init__(
        self,
        cameras: list[Camera],
        device: str = "cuda",
        verbose: bool = False,
        modalities: list[str] = ["rgbs", "masks"],
    ):
        """Create a tensor_reel object, containing all data
        stored contiguosly in tensors.

        Currently supports only static scenes, i.e. the first frame of each camera.

        Args:
            cameras (list): list of cameras objects
            device (str, optional): device to move tensors to. Defaults to "cuda".

        Attributes:
            rgbs (torch.Tensor): (N, T, H, W, 3) in [0, 1]
            masks (optional, torch.Tensor): (N, T, H, W, 1) in [0, 1] or None
            pose (torch.Tensor): (N, 4, 4)
            intrinsics_inv (torch.Tensor): (N, 3, 3)
        """

        if len(cameras) == 0:
            print_error("tensor reel has no cameras")

        data = {}
        c2w_all = []  # list of (4, 4) matrices
        w2c_all = []  # list of (4, 4) matrices
        intrinsics = []  # list of (3, 3) matrices
        intrinsics_inv = []  # list of (3, 3) matrices
        timestamps = []  # list of (T) tensors

        # collect data from all cameras
        pbar = tqdm(cameras, desc="tensor reel", ncols=100)
        for camera in pbar:
            # get camera data
            for key, val in camera.data.items():
                # populate data dict
                if key in modalities:
                    if key not in data:
                        data[key] = []
                    if val is not None:
                        data[key].append(torch.from_numpy(val))
                    else:
                        print_error(f"camera {camera.camera_idx} has no {key} data")

            # camera matrices
            c2w_all.append(torch.from_numpy(camera.get_pose()).float())
            w2c_all.append(torch.from_numpy(camera.get_pose_inv()).float())
            intrinsics.append(torch.from_numpy(camera.get_intrinsics()).float())
            intrinsics_inv.append(torch.from_numpy(camera.get_intrinsics_inv()).float())
            timestamps.append(torch.from_numpy(camera.get_timestamps()).float())

        # concat data and move to device
        for key, val in data.items():
            data[key] = torch.stack(val).to(device).contiguous()
        self.data = data

        # concat cameras matrices
        self.c2w_all = torch.stack(c2w_all).to(device).contiguous()  # (N, 4, 4)
        self.w2c_all = torch.stack(w2c_all).to(device).contiguous()  # (N, 4, 4)
        self.intrinsics = torch.stack(intrinsics).to(device).contiguous()  # (N, 3, 3)
        self.intrinsics_inv = (
            torch.stack(intrinsics_inv).to(device).contiguous()
        )  # (N, 3, 3)
        self.timestamps = torch.stack(timestamps).to(device).contiguous()  # (N, T)

        self.temporal_dim = cameras[0].get_temporal_dim()
        self.width, self.height = cameras[0].get_resolution()
        self.device = device

        if verbose:
            print_info(f"tensor reel on {self.device}")

    # TODO: deprecated
    # @torch.no_grad()
    # def get_next_cameras_batch(self, batch_size=8, cameras_idx=None, frames_idx=None):
    #     """Sample a batch of cameras from the tensor reel.

    #     Args:
    #         batch_size (int, optional): Defaults to 512.
    #         cameras_idx (torch.Tensor, optional): (N) Defaults to None.
    #         frames_idx (torch.Tensor, optional): (N) Defaults to None.

    #     Returns:
    #         cameras_idx (batch_size)
    #         projections (batch_size, 4, 4)
    #         rays_d (batch_size, height, width, 3)
    #         vals (dict):
    #             rgb (batch_size, height, width, 3)
    #             mask (batch_size, height, width, 1)
    #         frame_idx (batch_size)
    #     """

    #     # sample cameras_idx
    #     nr_cameras = self.c2w_all.shape[0]
    #     if cameras_idx is None:
    #         # sample among all cameras
    #         if nr_cameras < batch_size:
    #             # with repetitions
    #             cameras_idx = torch.randint(nr_cameras, (batch_size,))
    #         else:
    #             # without repetitions
    #             cameras_idx = torch.randperm(nr_cameras)[:batch_size]
    #     else:
    #         # sample among given cameras indices
    #         if len(cameras_idx) < batch_size:
    #             # sample with repetitions
    #             sampled_idx = torch.randint(len(cameras_idx), (batch_size,))
    #         else:
    #             # sample without repetitions
    #             sampled_idx = torch.randperm(len(cameras_idx), device=self.device)[
    #                 :batch_size
    #             ]
    #         cameras_idx = torch.tensor(cameras_idx, device=self.device)[sampled_idx]

    #     # sample frames_idx
    #     self.temporal_dim = self.rgbs.shape[1]
    #     if frames_idx is None:
    #         # sample among all frames with repetitions
    #         frames_idx = torch.randint(self.temporal_dim, (batch_size,))
    #     else:
    #         # sample among given frames indices with repetitions
    #         sampled_idx = torch.randint(len(frames_idx), (batch_size,))
    #         frames_idx = torch.tensor(frames_idx, device=self.device)[sampled_idx]

    #     vals = {}

    #     if self.rgbs is not None:
    #         rgbs = self.rgbs[cameras_idx, frames_idx]
    #         rgbs = image_uint8_to_float32(rgbs)
    #         vals["rgbs"] = rgbs

    #     if self.masks is not None:
    #         masks = self.masks[cameras_idx, frames_idx]
    #         masks = image_uint8_to_float32(masks)
    #         vals["masks"] = masks

    #     projections = self.projections[cameras_idx]

    #     # get rays_d for each pixel in each camera (batch)
    #     # get random pixels
    #     pixels = get_pixels(self.height, self.width, device=self.device)  # (H, W, 2)
    #     # get 2d points on the image plane
    #     jitter_pixels = False
    #     points_2d_screen = get_points_2d_screen_from_pixels(
    #         pixels, jitter_pixels
    #     )  # (N, 2)

    #     # unproject pixels to get view directions

    #     # pixels have height, width order, we need x, y, z order
    #     # points_2d_screen = points_2d_screen  # [..., [1, 0]]  # swap x and y
    #     points_2d_a_s = euclidean_to_homogeneous(points_2d_screen)
    #     points_2d_a_s = points_2d_a_s.unsqueeze(0)
    #     # print("points_2d_a_s", points_2d_a_s.shape)

    #     # from screen to camera coords
    #     intrinsics_inv_all = self.intrinsics_inv[cameras_idx]  # (batch_size, 3, 3)
    #     # print("intrinsics_inv_all", intrinsics_inv_all.shape)

    #     # from screen to camera coords
    #     points_3d_camera = torch.matmul(
    #         intrinsics_inv_all, points_2d_a_s.permute(0, 2, 1)
    #     ).permute(0, 2, 1)
    #     # print("points_3d_camera", points_3d_camera.shape)

    #     c2w_rot_all = self.c2w_all[cameras_idx, :3, :3]  # (batch_size, 3, 3)
    #     # print("c2w_rot_all", c2w_rot_all.shape)

    #     # rotate points to world space
    #     points_3d_world = torch.matmul(c2w_rot_all, points_3d_camera.permute(0, 2, 1)).permute(
    #         0, 2, 1
    #     )
    #     # print("points_3d_world", points_3d_world.shape)

    #     # normalize rays
    #     rays_d = torch.nn.functional.normalize(points_3d_world, dim=-1)

    #     # reshape
    #     view_dirs = rays_d.reshape(batch_size, self.height, self.width, 3)

    #     return cameras_idx, projections, view_dirs, vals, frames_idx

    @torch.no_grad()
    def get_next_rays_batch(
        self,
        batch_size: int = 512,
        cameras_idx: np.ndarray = None,
        frames_idx: np.ndarray = None,
        jitter_pixels: bool = False,
        nr_rays_per_pixel: int = 1,
    ):
        """Sample a batch of rays from the tensor reel.

        Args:
            batch_size (int, optional): Defaults to 512.
            cameras_idx (np.ndarray, optional): (N) Defaults to None.
            frames_idx (np.ndarray, optional): (N) Defaults to None.
            jitter_pixels (bool, optional): Defaults to False.
            nr_rays_per_pixel (int, optional): Defaults to 1.

        Returns:
            cameras_idx (np.ndarray): (batch_size)
            frames_idx (np.ndarray): (batch_size)
            rays_o (torch.Tensor): (batch_size, 3)
            rays_d (torch.Tensor): (batch_size, 3)
            vals (dict):
                rgb (torch.Tensor): (batch_size, H, W, 3)
                mask: (torch.Tensor): (batch_size, H, W, 1)
            timestamps (torch.Tensor): (batch_size)
        """

        assert nr_rays_per_pixel > 0, "nr_rays_per_pixel must be > 0"
        assert nr_rays_per_pixel == 1 or (
            nr_rays_per_pixel > 1 and jitter_pixels == True
        ), "jitter_pixels must be True if nr_rays_per_pixel > 1"

        real_batch_size = batch_size // nr_rays_per_pixel

        # sample cameras_idx
        nr_cameras = self.c2w_all.shape[0]
        if cameras_idx is None:
            # Sample among all cameras with repetitions
            cameras_idx = np.random.randint(0, nr_cameras, size=real_batch_size)
        else:
            # Sample among given camera indices with repetitions
            sampled_idx = np.random.randint(0, len(cameras_idx), size=real_batch_size)
            cameras_idx = cameras_idx[sampled_idx]
        cameras_idx = cameras_idx.astype(np.int32)

        # sample frames_idx
        if frames_idx is None:
            # Sample among all frames with repetitions
            frames_idx = np.random.randint(0, self.temporal_dim, size=real_batch_size)
        else:
            # Sample among given frame indices with repetitions
            sampled_idx = np.random.randint(0, len(frames_idx), size=real_batch_size)
            frames_idx = frames_idx[sampled_idx]
        frames_idx = frames_idx.astype(np.int32)

        # get random pixels
        pixels = get_random_pixels(
            self.height, self.width, real_batch_size, device=self.device
        )  # (N, 2)

        # repeat pixels if needed
        if nr_rays_per_pixel > 1:
            # torch Tensors
            pixels = pixels.repeat_interleave(nr_rays_per_pixel, dim=0)  # (N, 2)
            # numpy arrays
            cameras_idx = np.repeat(cameras_idx, nr_rays_per_pixel, axis=0)  # (N)
            frames_idx = np.repeat(frames_idx, nr_rays_per_pixel, axis=0)  # (N)

        # get 2d points on the image plane
        points_2d_screen = get_points_2d_screen_from_pixels(
            pixels, jitter_pixels
        )  # (N, 2)

        # get ground truth rgbs values at pixels
        vals = get_data_per_points_2d_screen(
            points_2d_screen=points_2d_screen,
            cameras_idx=cameras_idx,
            frames_idx=frames_idx,
            data_dict=self.data,
        )

        # get a ray for each pixel in corresponding camera frame
        rays_o, rays_d = get_rays_per_points_2d_screen(
            c2w=self.c2w_all[cameras_idx],
            intrinsics_inv=self.intrinsics_inv[cameras_idx],
            points_2d_screen=points_2d_screen,
        )

        # timestamps
        timestamps = self.timestamps[cameras_idx, frames_idx]  # (N)

        return {
            "cameras_idx": cameras_idx,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "vals": vals,
            "timestamps": timestamps,
            "frames_idx": frames_idx,
        }

    # TODO: deprecated
    # def get_cameras_rays_per_points_2d(c2w_all, intrinsics_inv_all, points_2d_screen):
    #     """given a list of c2w, intrinsics_inv and points_2d_screen, return rays origins and
    #     directions from multiple cameras

    #     args:
    #         c2w_all (torch.Tensor): (N, 4, 4)
    #         intrinsics_inv_all (torch.Tensor): (N, 3, 3)
    #         points_2d_screen (torch.Tensor, int): (N, 2) with values in [0, W-1], [0, H-1]

    #     out:
    #         rays_o (torch.Tensor): (N, 3)
    #         rays_d (torch.Tensor): (N, 3)
    #     """

    #     assert c2w_all.ndim == 3 and c2w_all.shape[-2:] == (
    #         4,
    #         4,
    #     ), "c2w_all must be (N, 4, 4)"
    #     assert intrinsics_inv_all.ndim == 3 and intrinsics_inv_all.shape[-2:] == (
    #         3,
    #         3,
    #     ), "intrinsics_inv_all must be (N, 3, 3)"
    #     assert (
    #         points_2d_screen.ndim == 2 and points_2d_screen.shape[-1] == 2
    #     ), "points_2d_screen must be (N, 2)"
    #     assert (
    #         c2w_all.shape[0] == intrinsics_inv_all.shape[0] == points_2d_screen.shape[0]
    #     ), "c2w_all, intrinsics_inv_all and points_2d_screen must have the same batch size"

    #     # ray origin are the cameras centers
    #     rays_o = c2w_all[:, :3, -1]

    #     # unproject pixels to get view directions

    #     # pixels have height, width order, we need x, y, z order
    #     points_2d_a_s = euclidean_to_homogeneous(points_2d_screen)

    #     # from screen to camera coords (out is (N, 3, 1))
    #     points_3d_camera = intrinsics_inv_all @ points_2d_a_s.unsqueeze(-1)

    #     # rotate points to world space
    #     points_3d_world = c2w_all[:, :3, :3] @ points_3d_camera
    #     points_3d_world = points_3d_world.reshape(-1, 3)

    #     # normalize rays
    #     rays_d = torch.nn.functional.normalize(points_3d_world, dim=-1)

    #     return rays_o, rays_d

    # TODO: deprecated
    # def get_data_per_points_2d_screen(
    #     points_2d_screen, cameras_idx, frames_idx, rgbs=None, masks=None
    # ):
    #     """given a list of 2d points on the image plane and a list of rgbs,
    #     return rgb and mask values at points_2d_screen

    #     args:
    #         points_2d_screen (torch.Tensor, int): (N, 2) values in [0, W-1], [0, H-1].
    #         cameras_idx (torch.Tensor): (N) camera indices
    #         frames_idx (torch.Tensor): (N) frame indices.
    #         rgbs (optional, torch.Tensor, uint8): (N, T, H, W, 3) or None
    #         masks (optional, torch.Tensor, uint8): (N, T, H, W, 1) or None
    #     out:
    #         vals (dict):
    #             rgb_vals (optional, torch.Tensor, float32): (N, 3)
    #             mask_vals (optional, torch.Tensor, float32): (N, 1)
    #     """

    #     assert points_2d_screen.shape[1] == 2, "points_2d_screen must be (N, 2)"

    #     wh = points_2d_screen.int()  # floor to get pixels
    #     hw = wh[:, [1, 0]]  # invert y, x to x, y
    #     i, j = hw[:, 0], hw[:, 1]

    #     # prepare output
    #     vals = {}

    #     # rgb
    #     rgb_vals = None
    #     if rgbs is not None:
    #         rgb_vals = rgbs[cameras_idx, frames_idx, i, j]
    #         rgb_vals = image_uint8_to_float32(rgb_vals)
    #         vals["rgbs"] = rgb_vals

    #     # mask
    #     mask_vals = None
    #     if masks is not None:
    #         mask_vals = masks[cameras_idx, frames_idx, i, j]
    #         mask_vals = image_uint8_to_float32(mask_vals)
    #         vals["masks"] = mask_vals

    #     # TODO: get other frame data

    #     assert rgb_vals is None or rgb_vals.shape[1] == 3, "rgb must be (N, 3)"
    #     assert mask_vals is None or mask_vals.shape[1] == 1, "mask_vals must be (N, 1)"

    #     return vals

    def __str__(self) -> str:
        string = "\nTensorReel\n"
        string += f"device: {self.device}\n"
        string += f"c2w_all: {self.c2w_all.shape}, {self.c2w_all.dtype}\n"
        string += f"intrinsics_inv: {self.intrinsics_inv.shape}, {self.intrinsics_inv.dtype}\n"
        # string += f"projections: {self.projections.shape}, {self.projections.dtype}\n"
        for key, val in self.data.items():
            string += f"{key}: {val.shape}, {val.dtype}\n"
        return string
