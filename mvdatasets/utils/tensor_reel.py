import torch
from tqdm import tqdm

from mvdatasets.utils.raycasting import (
    get_random_pixels,
    get_cameras_rays_per_points_2d,
    get_tensor_reel_frames_per_pixels,
    get_points_2d_from_pixels,
    get_random_pixels_from_error_map
)


class TensorReel:
    def __init__(self, cameras_list, width=None, height=None, device="cuda", verbose=False):
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
        if verbose:
            pbar = tqdm(cameras_list, desc="tensor reel", ncols=100)
        else:
            pbar = cameras_list
        for camera in pbar:
            
            # rgbs
            if camera.has_rgbs():
                rgbs.append(torch.from_numpy(camera.get_rgbs()))
            
            # masks
            if camera.has_masks():
                masks.append(torch.from_numpy(camera.get_masks()))
            
            # normals
            if camera.has_normals():
                normals.append(torch.from_numpy(camera.get_normals()))
            
            # depths
            if camera.has_depths():
                depths.append(torch.from_numpy(camera.get_depths()))
            
            # instance_masks
            if camera.has_instance_masks():
                instance_masks.append(torch.from_numpy(camera.get_instance_masks()))
            
            # semantic_masks
            if camera.has_semantic_masks():
                semantic_masks.append(torch.from_numpy(camera.get_semantic_masks()))
            
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
            if verbose:
                print("moving tensor reel to device")
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
        else:
            if verbose:
                print("tensor reel on cpu")

        self.device = device
        if self.rgbs is not None:
            self.height = self.rgbs.shape[2]
            self.width = self.rgbs.shape[3]
        else:
            assert height is not None and width is not None, "height and width must be specified if cameras have not rgbs"
            self.height = height
            self.width = width

    @torch.no_grad()
    def get_next_batch(
        self, batch_size=512, cameras_idx=None, frame_idx=None, jitter_pixels=False, nr_rays_per_pixel=1, masked_sampling=False
    ):
        assert nr_rays_per_pixel > 0, "nr_rays_per_pixel must be > 0"
        # assert batch_size % nr_rays_per_pixel == 0, "batch_size must be a multiple of nr_rays_per_pixel"
        assert nr_rays_per_pixel == 1 or (nr_rays_per_pixel > 1 and jitter_pixels == True), "jitter_pixels must be True if nr_rays_per_pixel > 1"
        
        real_batch_size = batch_size // nr_rays_per_pixel
        
        # random int in range [0, nr_cameras-1] with repetitions
        nr_cameras = self.poses.shape[0]  # alway sample from all cameras
        if cameras_idx is None:
            camera_idx = torch.randint(nr_cameras, (real_batch_size,))
        else:
            # sample among given cameras indices with repetitions
            sampled_idx = torch.randint(len(cameras_idx), (real_batch_size,))
            camera_idx = torch.tensor(cameras_idx, device=self.device)[sampled_idx]

        # TODO: improve
        # random int in range [0, nr_frames_in_sequence-1] with repetitions
        if frame_idx is None:
            # if self.rgbs is not None:
            #     nr_frames_in_sequence = self.rgbs.shape[1]
            # else: 
            #     nr_frames_in_sequence = 1
            # frame_idx = torch.randint(nr_frames_in_sequence, (real_batch_size,))
            frame_idx = torch.zeros(real_batch_size).int()
        else:
            frame_idx = (torch.ones(real_batch_size) * frame_idx).int()
        # frame_idx = frame_idx.repeat_interleave(nr_rays_per_pixel, dim=0)
        
        # if masked_sampling:
        #     # TODO: test if correct and performance drop
        #     for idx in camera_idx:
        #         nr_rays = int((real_batch_size / len(camera_idx)) / nr_rays_per_pixel)
        #         print("nr rays per camera", nr_rays)
                
        #         pixels = get_random_pixels_from_error_map(
        #             self.masks[idx], self.height, self.width, nr_rays, device=self.device
        #         )
        #         print("pixels.shape", pixels.shape)
                
        #         # repeat pixels nr_rays_per_pixel times
        #         pixels = pixels.repeat_interleave(nr_rays_per_pixel, dim=0)
        #         print("pixels.shape", pixels.shape)
                
        #         # get 2d points on the image plane
        #         points_2d = get_points_2d_from_pixels(pixels, jitter_pixels, self.height, self.width)
        #         print("points_2d.shape", points_2d.shape)
                
        #         # get a ray for each pixel in corresponding camera frame
        #         rays_o, rays_d = get_cameras_rays_per_points_2d(
        #             c2w_all=self.poses[idx].repeat(nr_rays),
        #             intrinsics_inv_all=self.intrinsics_inv[idx].repeat(nr_rays),
        #             points_2d_screen=points_2d
        #         )
        #         # get ground truth rgbs values at pixels
        #         vals = get_tensor_reel_frames_per_pixels(
        #             points_2d=points_2d,
        #             camera_idx=idx,
        #             frame_idx=frame_idx,
        #             rgbs=self.rgbs,
        #             masks=self.masks
        #         )
            
        # else:
        
        # get random pixels
        pixels = get_random_pixels(
            self.height, self.width, real_batch_size, device=self.device
        )
        # get ground truth rgbs values at pixels
        vals = get_tensor_reel_frames_per_pixels(
            pixels=pixels,
            camera_idx=camera_idx,
            frame_idx=frame_idx,
            rgbs=self.rgbs,
            masks=self.masks
        )
        # get 2d points on the image plane
        pixels = pixels.repeat_interleave(nr_rays_per_pixel, dim=0)
        # print("pixels", pixels)
        points_2d = get_points_2d_from_pixels(pixels, jitter_pixels, self.height, self.width)

        # get a ray for each pixel in corresponding camera frame
        camera_idx = camera_idx.repeat_interleave(nr_rays_per_pixel, dim=0)
        # print("camera_idx", camera_idx)
        rays_o, rays_d = get_cameras_rays_per_points_2d(
            c2w_all=self.poses[camera_idx],
            intrinsics_inv_all=self.intrinsics_inv[camera_idx],
            points_2d_screen=points_2d
        )

        return camera_idx, rays_o, rays_d, vals, frame_idx
    
    def __str__(self) -> str:
        string = "\nTensorReel\n"
        string += f"device: {self.device}\n"
        string += f"poses: {self.poses.shape}, {self.poses.dtype}\n"
        string += f"intrinsics_inv: {self.intrinsics_inv.shape}, {self.intrinsics_inv.dtype}\n"
        if self.rgbs is not None:
            string += f"rgbs: {self.rgbs.shape}, {self.rgbs.dtype}\n"
        if self.masks is not None:
            string += f"masks: {self.masks.shape}, {self.masks.dtype}\n"
        if self.normals is not None:
            string += f"normals: {self.normals.shape}, {self.normals.dtype}\n"
        if self.depths is not None:
            string += f"depths: {self.depths.shape}, {self.depths.dtype}\n"
        if self.instance_masks is not None:
            string += f"instance_masks: {self.instance_masks.shape}, {self.instance_masks.dtype}\n"
        if self.semantic_masks is not None:
            string += f"semantic_masks: {self.semantic_masks.shape}, {self.semantic_masks.dtype}\n"
        return string