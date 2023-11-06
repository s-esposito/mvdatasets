import torch

from mvdatasets.utils.raycasting import (
    get_cameras_rays_per_pixel,
    get_random_pixels,
    get_cameras_frames_per_pixels,
)


class TensorReel:
    def __init__(self, cameras_list, device="cuda"):
        """Create a tensor_reel object, containing all data
        stored contiguosly in tensors.

        Args:
            cameras_list (list): list of cameras objects
            device (str, optional): device to move tensors to. Defaults to "cuda".

        Attributes:
            frames (torch.tensor): (N, T, H, W, 3) in [0, 1]
            masks (optional, torch.tensor): (N, T, H, W, 1) in [0, 1] or None
            pose (torch.tensor): (N, 4, 4)
            intrinsics_inv (torch.tensor): (N, 3, 3)
        """

        frames = []
        masks = []
        poses = []
        intrinsics_inv = []

        # collect data from all cameras
        for camera in cameras_list:
            frames.append(torch.from_numpy(camera.get_frames()).float())
            if camera.has_masks:
                masks.append(torch.from_numpy(camera.get_masks()).float())
            poses.append(torch.from_numpy(camera.get_pose()).float())
            intrinsics_inv.append(torch.from_numpy(camera.get_intrinsics_inv()).float())

        # concat camera data in big tensors
        self.frames = torch.stack(frames)
        if len(masks) > 0:
            self.masks = torch.stack(masks)
        else:
            self.masks = None
        self.poses = torch.stack(poses)
        self.intrinsics_inv = torch.stack(intrinsics_inv)

        # move tensors to desired device
        if device != "cpu":
            self.intrinsics_inv = self.intrinsics_inv.to(device)
            self.poses = self.poses.to(device)
            self.frames = self.frames.to(device)
            if self.masks is not None:
                self.masks = self.masks.to(device)

        self.device = device
        self.height = self.frames.shape[2]
        self.width = self.frames.shape[3]

    def get_next_batch(self, batch_size=512, cameras_idxs=None, timestamp=None):
        # random int in range [0, nr_cameras-1] with repetitions
        nr_cameras = self.poses.shape[0]  # alway sample from all cameras
        if cameras_idxs is None:
            camera_idx = torch.randint(nr_cameras, (batch_size,))
        else:
            # sample from given cameras idxs with repetitions
            sampled_idx = torch.randint(len(cameras_idxs), (batch_size,))
            camera_idx = torch.tensor(cameras_idxs, device=self.device)[sampled_idx]

        # random int in range [0, nr_frames_in_sequence-1] with repetitions
        nr_frames_in_sequence = self.frames.shape[1]
        if timestamp is None:
            frame_idx = torch.randint(nr_frames_in_sequence, (batch_size,))
        else:
            frame_idx = (torch.ones(batch_size) * timestamp).int()

        # get random pixels
        pixels = get_random_pixels(
            self.height, self.width, batch_size, device=self.device
        )

        # get a ray for each pixel in corresponding camera frame
        rays_o, rays_d = get_cameras_rays_per_pixel(
            self.poses[camera_idx], self.intrinsics_inv[camera_idx], pixels
        )

        # get rgb and mask gt values at pixels
        gt_rgb, gt_mask = get_cameras_frames_per_pixels(
            camera_idx, frame_idx, pixels, self.frames, self.masks
        )

        return camera_idx, rays_o, rays_d, gt_rgb, gt_mask, frame_idx
