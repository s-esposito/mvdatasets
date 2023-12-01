import torch

from mvdatasets.utils.raycasting import (
    get_random_pixels,
    get_cameras_rays_per_points_2d,
    get_cameras_frames_per_points_2d,
    get_points_2d_from_pixels
)


class TensorReel:
    def __init__(self, cameras_list, device="cuda"):
        """Create a tensor_reel object, containing all data
        stored contiguosly in tensors.
        
        Currently supports only static scenes, i.e. the first frame of each camera.

        Args:
            cameras_list (list): list of cameras objects
            device (str, optional): device to move tensors to. Defaults to "cuda".

        Attributes:
            rgbs (torch.tensor): (N, T, H, W, 3) in [0, 1]
            masks (optional, torch.tensor): (N, T, H, W, 1) in [0, 1] or None
            pose (torch.tensor): (N, 4, 4)
            intrinsics_inv (torch.tensor): (N, 3, 3)
        """

        # modalities
        rgbs = []
        masks = []
        normals = []
        depths = []
        instance_masks = []
        semantic_masks = []
        
        poses = []
        intrinsics_inv = []

        # collect data from all cameras
        for camera in cameras_list:
            
            # rgbs
            if camera.has_rgbs():
                rgbs.append(torch.from_numpy(camera.get_rgbs()).float())
            
            # masks
            if camera.has_masks():
                masks.append(torch.from_numpy(camera.get_masks()).float())
            
            # normals
            if camera.has_normals:
                normals.append(torch.from_numpy(camera.get_normals()).float())
            
            # depths
            if camera.has_depths:
                depths.append(torch.from_numpy(camera.get_depths()).float())
            
            # instance_masks
            if camera.has_instance_masks:
                instance_masks.append(torch.from_numpy(camera.get_instance_masks()).float())
            
            # semantic_masks
            if camera.has_semantic_masks:
                semantic_masks.append(torch.from_numpy(camera.get_semantic_masks()).float())
            
            # camera params
            poses.append(torch.from_numpy(camera.get_pose()).float())
            intrinsics_inv.append(torch.from_numpy(camera.get_intrinsics_inv()).float())

        # concat rgb
        if len(rgbs) > 0:
            self.rgbs = torch.stack(rgbs)
        else:
            self.rgbs = None
        
        # concat masks
        if len(masks) > 0:
            self.masks = torch.stack(masks)
        else:
            self.masks = None
        
        # concat normals
        if len(normals) > 0:
            self.normals = torch.stack(normals)
        else:
            self.normals = None
        
        # concat depths
        if len(depths) > 0:
            self.depths = torch.stack(depths)
        else:
            self.depths = None
        
        # concat instance_masks
        if len(instance_masks) > 0:
            self.instance_masks = torch.stack(instance_masks)
        else:
            self.instance_masks = None
            
        # concat semantic_masks
        if len(semantic_masks) > 0:
            self.semantic_masks = torch.stack(semantic_masks)
        else:
            self.semantic_masks = None
        
        # concat camera params
        self.poses = torch.stack(poses)
        self.intrinsics_inv = torch.stack(intrinsics_inv)

        # move tensors to desired device
        if device != "cpu":
            self.intrinsics_inv = self.intrinsics_inv.to(device)
            self.poses = self.poses.to(device)
            if self.rgbs is not None:
                self.rgbs = self.rgbs.to(device)
            if self.masks is not None:
                self.masks = self.masks.to(device)
            if self.normals is not None:
                self.normals = self.normals.to(device)
            if self.depths is not None:
                self.depths = self.depths.to(device)
            if self.instance_masks is not None:
                self.instance_masks = self.instance_masks.to(device)
            if self.semantic_masks is not None:
                self.semantic_masks = self.semantic_masks.to(device)

        self.device = device
        self.height = self.rgbs.shape[2]
        self.width = self.rgbs.shape[3]

    def get_next_batch(
        self, batch_size=512, cameras_idx=None, frame_idx=None, jitter_pixels=False
    ):
        # random int in range [0, nr_cameras-1] with repetitions
        nr_cameras = self.poses.shape[0]  # alway sample from all cameras
        if cameras_idx is None:
            camera_idx = torch.randint(nr_cameras, (batch_size,))
        else:
            # sample from given cameras idxs with repetitions
            sampled_idx = torch.randint(len(cameras_idx), (batch_size,))
            camera_idx = torch.tensor(cameras_idx, device=self.device)[sampled_idx]

        # random int in range [0, nr_frames_in_sequence-1] with repetitions
        nr_frames_in_sequence = self.rgbs.shape[1]
        if frame_idx is None:
            frame_idx = torch.randint(nr_frames_in_sequence, (batch_size,))
        else:
            frame_idx = (torch.ones(batch_size) * frame_idx).int()

        # get random pixels
        pixels = get_random_pixels(
            self.height, self.width, batch_size, device=self.device
        )
        
        points_2d = get_points_2d_from_pixels(pixels, jitter_pixels)

        # get a ray for each pixel in corresponding camera frame
        rays_o, rays_d = get_cameras_rays_per_points_2d(
            self.poses[camera_idx],
            self.intrinsics_inv[camera_idx],
            points_2d
        )

        # get ground truth rgbs values at pixels
        frame_vals = get_cameras_frames_per_points_2d(
            pixels, camera_idx, frame_idx, rgbs=self.rgbs, masks=self.masks
        )

        return camera_idx, rays_o, rays_d, frame_vals, frame_idx
