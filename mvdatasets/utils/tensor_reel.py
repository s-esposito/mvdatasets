import torch

from mvdatasets.utils.raycasting import (
    get_cameras_rays_per_pixel,
    get_random_pixels,
    get_cameras_frames_per_pixels,
)


class TensorReel:
    def __init__(self, dataset, device="cuda"):
        intrinsics_inv = []
        poses = []
        frames = []
        masks = []
        # frames_dims = []

        # for all cameras in dataset
        for camera in dataset.cameras:
            intrinsics_inv.append(camera.get_intrinsics_inv())
            poses.append(camera.get_pose())
            frames.append(camera.get_frames())
            if camera.get_masks() is not None:
                masks.append(camera.get_masks())
            # frames_dims.append((camera.height, camera.width))

        # concat camera data to single tensors
        self.intrinsics_inv = torch.stack(intrinsics_inv)
        self.poses = torch.stack(poses)
        self.frames = torch.stack(frames)
        if len(masks) > 0:
            self.masks = torch.stack(masks)
        else:
            self.masks = None

        # move tensors to device
        if device != "cpu":
            self.intrinsics_inv = self.intrinsics_inv.to(device)
            self.poses = self.poses.to(device)
            self.frames = self.frames.to(device)
            if self.masks is not None:
                self.masks = self.masks.to(device)

        # print("intrinsics_inv", self.intrinsics_inv.shape, self.intrinsics_inv.device)
        # print("poses", self.poses.shape, self.poses.device)
        # print("frames", self.frames.shape, self.frames.device)
        # print("masks", self.masks.shape, self.masks.device)

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
            # print("sampled_idx", sampled_idx)
            camera_idx = torch.tensor(cameras_idxs, device=self.device)[sampled_idx]
            # print("camera_idx", camera_idx)
            # camera_idx = (torch.ones(batch_size) * nr_camera).int()

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
