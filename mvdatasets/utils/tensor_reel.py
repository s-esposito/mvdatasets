import torch
import numpy as np
from tqdm import tqdm

from mvdatasets.utils.images import image_uint8_to_float32

from mvdatasets.utils.raycasting import (
    get_random_pixels,
    get_cameras_rays_per_points_2d,
    get_tensor_reel_frames_per_pixels,
    get_points_2d_from_pixels,
    get_random_pixels_from_error_map
)


class TensorReel:
    def __init__(
        self,
        cameras_list,
        width=None,
        height=None,
        opengl_standard=False,
        near=0.1,
        far=1000.0,
        device="cuda",
        verbose=False
    ):
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
        projections = []

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
            
            # camera matrices
            poses.append(torch.from_numpy(camera.get_pose()).float())
            intrinsics_inv.append(torch.from_numpy(camera.get_intrinsics_inv()).float())
            proj = camera.get_projection(
                opengl_standard=opengl_standard,
                near=near,
                far=far
            )
            projections.append(torch.from_numpy(proj).float())

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
        
        # concat cameras matrices
        self.poses = torch.stack(poses)
        self.intrinsics_inv = torch.stack(intrinsics_inv)
        self.projections = torch.stack(projections)

        # move tensors to desired device
        if device != "cpu":
            if verbose:
                print("moving tensor reel to device")
            self.intrinsics_inv = self.intrinsics_inv.to(device)
            self.poses = self.poses.to(device)
            self.projections = self.projections.to(device)
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
    def get_next_cameras_batch(
        self,
        batch_size=8,
        cameras_idx=None,
        frames_idx=None
    ):
        """Sample a batch of cameras from the tensor reel.

        Args:
            batch_size (int, optional): Defaults to 512.
            cameras_idx (torch.tensor, optional): (N) Defaults to None.
            frames_idx (torch.tensor, optional): (N) Defaults to None.

        Returns:
            cameras_idx (batch_size)
            projections (batch_size, 4, 4)
            vals (dict):
                rgb (batch_size, height, width, 3)
                mask (batch_size, height, width, 1)
            frame_idx (batch_size)
        """
        
        # sample cameras_idx
        nr_cameras = self.poses.shape[0]
        if cameras_idx is None:
            # sample among all cameras
            if nr_cameras < batch_size:
                # with repetitions
                cameras_idx = torch.randint(nr_cameras, (batch_size,))
            else:
                # without repetitions
                cameras_idx = torch.randperm(nr_cameras)[:batch_size]
        else:
            # sample among given cameras indices
            if len(cameras_idx) < batch_size:
                # sample with repetitions
                sampled_idx = torch.randint(len(cameras_idx), (batch_size,))
            else:
                # sample without repetitions
                sampled_idx = torch.tensor(len(cameras_idx), device=self.device)[:batch_size]
            cameras_idx = torch.tensor(cameras_idx, device=self.device)[sampled_idx]
            
        # TODO: sample frames_idx for each camera (if not given)
        frames_idx = torch.zeros(batch_size).int()
        # if frames_idx is None:
        #   ....
        
        vals = {}
        
        if self.rgbs is not None:
            rgbs = self.rgbs[cameras_idx, frames_idx]
            rgbs = image_uint8_to_float32(rgbs)
            vals["rgb"] = rgbs
        
        if self.masks is not None:
            masks = self.masks[cameras_idx, frames_idx]
            masks = image_uint8_to_float32(masks)
            vals["mask"] = masks
            
        projections = self.projections[cameras_idx]

        return cameras_idx, projections, vals, frames_idx
    
    @torch.no_grad()
    def get_next_rays_batch(
        self,
        batch_size=512,
        cameras_idx=None,
        frames_idx=None,
        jitter_pixels=False,
        nr_rays_per_pixel=1,
        # masked_sampling=False
    ):
        """Sample a batch of rays from the tensor reel.

        Args:
            batch_size (int, optional): Defaults to 512.
            cameras_idx (torch.tensor, optional): (N) Defaults to None.
            frames_idx (torch.tensor, optional): (N) Defaults to None.
            jitter_pixels (bool, optional): Defaults to False.
            nr_rays_per_pixel (int, optional): Defaults to 1.

        Returns:
            cameras_idx (batch_size)
            rays_o (batch_size, 3)
            rays_d (batch_size, 3)
            vals (dict):
                rgb (batch_size, 3)
                mask (batch_size, 1)
            frame_idx (batch_size)
        """
        
        assert nr_rays_per_pixel > 0, "nr_rays_per_pixel must be > 0"
        assert nr_rays_per_pixel == 1 or (nr_rays_per_pixel > 1 and jitter_pixels == True), "jitter_pixels must be True if nr_rays_per_pixel > 1"
        
        real_batch_size = batch_size // nr_rays_per_pixel
        
        # sample cameras_idx
        nr_cameras = self.poses.shape[0]
        if cameras_idx is None:
            # sample among all cameras with repetitions
            cameras_idx = torch.randint(nr_cameras, (real_batch_size,))
        else:
            # sample among given cameras indices with repetitions
            sampled_idx = torch.randint(len(cameras_idx), (real_batch_size,))
            cameras_idx = torch.tensor(cameras_idx, device=self.device)[sampled_idx]

        # TODO: sample frames_idx for each camera (if not given)
        frames_idx = torch.zeros(real_batch_size).int()
        
        # get random pixels
        pixels = get_random_pixels(
            self.height,
            self.width,
            real_batch_size,
            device=self.device
        )
        # get ground truth rgbs values at pixels
        vals = get_tensor_reel_frames_per_pixels(
            pixels=pixels,
            cameras_idx=cameras_idx,
            frames_idx=frames_idx,
            rgbs=self.rgbs,
            masks=self.masks
        )
        # get 2d points on the image plane
        pixels = pixels.repeat_interleave(nr_rays_per_pixel, dim=0)
        points_2d = get_points_2d_from_pixels(
            pixels,
            jitter_pixels,
            self.height,
            self.width
        )

        # get a ray for each pixel in corresponding camera frame
        cameras_idx = cameras_idx.repeat_interleave(nr_rays_per_pixel, dim=0)
        rays_o, rays_d = get_cameras_rays_per_points_2d(
            c2w_all=self.poses[cameras_idx],
            intrinsics_inv_all=self.intrinsics_inv[cameras_idx],
            points_2d_screen=points_2d
        )
        
        # random int in range [0, nr_frames_in_sequence-1] with repetitions
        # if frames_idx is None:
            # if self.rgbs is not None:
            #     nr_frames_in_sequence = self.rgbs.shape[1]
            # else: 
            #     nr_frames_in_sequence = 1
            # frames_idx = torch.randint(nr_frames_in_sequence, (real_batch_size,))
            
        # else:
        #    frames_idx = (torch.ones(real_batch_size) * frames_idx).int()
        # frames_idx = frames_idx.repeat_interleave(nr_rays_per_pixel, dim=0)
        
        # if masked_sampling:
        #     for idx in cameras_idx:
        #         nr_rays = int((real_batch_size / len(cameras_idx)) / nr_rays_per_pixel)
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
        #             cameras_idx=idx,
        #             frames_idx=frames_idx,
        #             rgbs=self.rgbs,
        #             masks=self.masks
        #         )

        return cameras_idx, rays_o, rays_d, vals, frames_idx
    
    def __str__(self) -> str:
        string = "\nTensorReel\n"
        string += f"device: {self.device}\n"
        string += f"poses: {self.poses.shape}, {self.poses.dtype}\n"
        string += f"intrinsics_inv: {self.intrinsics_inv.shape}, {self.intrinsics_inv.dtype}\n"
        string += f"projections: {self.projections.shape}, {self.projections.dtype}\n"
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