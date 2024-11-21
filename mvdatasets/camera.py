import numpy as np
import torch
import cv2 as cv
from typing import Union, Tuple

from mvdatasets.geometry.common import (
    apply_transformation_3d,
    global_perspective_projection,
    global_inv_perspective_projection,
    opengl_projection_matrix_from_intrinsics,
    opengl_matrix_world_from_w2c,
    get_mask_points_out_image,
)
from mvdatasets.utils.printing import print_error, print_warning
from mvdatasets.utils.raycasting import (
    get_pixels,
    get_points_2d_screen_from_pixels,
    get_random_pixels,
    get_rays_per_points_2d_screen,
    get_data_per_points_2d_screen,
)


class Camera:
    """Camera class to manage intrinsics, pose, and various image data.

    Assumptions:
    - All data have the same temporal and spatial dimensions.
    """

    def __init__(
        self,
        intrinsics: np.ndarray,
        pose: np.ndarray,
        rgbs: np.ndarray = None,
        masks: np.ndarray = None,
        normals: np.ndarray = None,
        depths: np.ndarray = None,
        instance_masks: np.ndarray = None,
        semantic_masks: np.ndarray = None,
        global_transform: np.ndarray = None,
        local_transform: np.ndarray = None,
        camera_idx: int = 0,
        width: int = None,
        height: int = None,
        temporal_dim: int = 1,
        subsample_factor: int = 1,
        verbose: bool = False,
    ):
        """
        Initialize a Camera object.

        Args:
            intrinsics (np.ndarray): (3, 3) Camera intrinsic matrix (camtopix).
            pose (np.ndarray): (4, 4) Camera extrinsic matrix (camera-to-world transformation).
            rgbs (np.ndarray, optional): (T, H, W, 3) RGB images, uint8 or float32.
            masks (np.ndarray, optional): (T, H, W, 1) Binary masks, uint8 or float32.
            normals (np.ndarray, optional): (T, H, W, 3) Surface normals, uint8 or float32.
            depths (np.ndarray, optional): (T, H, W, 1) Depth maps, uint8 or float32.
            instance_masks (np.ndarray, optional): (T, H, W, 1) Instance segmentation masks, uint8.
            semantic_masks (np.ndarray, optional): (T, H, W, 1) Semantic segmentation masks, uint8.
            global_transform (np.ndarray, optional): Global transformation matrix.
            local_transform (np.ndarray, optional): Local transformation matrix.
            camera_idx (int, optional): Camera index. Defaults to 0.
            width (int, optional): Image width, required if no images are provided.
            height (int, optional): Image height, required if no images are provided.
            subsample_factor (int, optional): Subsampling factor for images. Defaults to 1.

        Raises:
            ValueError: If both images and width/height are missing.
        """

        self.verbose = verbose

        # Validate input dimensions
        assert intrinsics.shape == (3, 3), "`intrinsics` must be a 3x3 matrix."
        assert pose.shape == (4, 4), "`pose` must be a 4x4 matrix."
        self.intrinsics = intrinsics
        self.intrinsics_inv = np.linalg.inv(intrinsics)
        self.pose = pose
        self.camera_idx = camera_idx

        # assert shapes are correct
        if rgbs is not None:
            assert rgbs.ndim == 4 and rgbs.shape[-1] == 3
        if masks is not None:
            assert masks.ndim == 4 and masks.shape[-1] == 1
        if depths is not None:
            assert depths.ndim == 4 and depths.shape[-1] == 1

        # Store image-based data
        self.data = {
            "rgbs": rgbs,
            "masks": masks,
            "normals": normals,
            "depths": depths,
            "instance_masks": instance_masks,
            "semantic_masks": semantic_masks,
        }

        # Infer dimensions from provided images
        if rgbs is not None:
            self.temporal_dim, self.height, self.width = rgbs.shape[:3]
        elif width is not None and height is not None:
            self.height = height
            self.width = width
            self.temporal_dim = temporal_dim
        else:
            raise ValueError("Either provide `rgbs` or specify `width` and `height`.")

        # Validate dimensions of all data
        self._validate_data()

        # transforms
        if global_transform is not None:
            self.global_transform = global_transform
        else:
            self.global_transform = np.eye(4)

        if local_transform is not None:
            self.local_transform = local_transform
        else:
            self.local_transform = np.eye(4)

        # Subsample data if needed
        if subsample_factor > 1:
            self.resize(subsample_factor)

        if verbose:
            print(self.__str__())

    def _validate_data(self) -> None:
        """Validate that all data have consistent dimensions."""
        for key, modality in self.data.items():
            if modality is None:
                continue

            # Validate temporal and spatial dimensions
            t, h, w = modality.shape[:3]
            if self.temporal_dim is not None and t != self.temporal_dim:
                raise ValueError(
                    f"Modality `{key}` has inconsistent temporal dimension: {t} (expected {self.temporal_dim})."
                )
            if h != self.height or w != self.width:
                raise ValueError(
                    f"Modality `{key}` has inconsistent spatial dimensions: ({h}, {w}) (expected ({self.height}, {self.width}))."
                )

            # Validate channel dimensions if applicable
            if key in ["rgbs", "normals"] and modality.shape[-1] != 3:
                raise ValueError(
                    f"Modality `{key}` must have 3 channels; found {modality.shape[-1]}."
                )
            if (
                key in ["masks", "depths", "instance_masks", "semantic_masks"]
                and modality.shape[-1] != 1
            ):
                raise ValueError(
                    f"Modality `{key}` must have 1 channel; found {modality.shape[-1]}."
                )

    def _scale_intrinsics(self, s_width: float, s_height: float) -> None:
        """scales the intrinsics matrix"""
        self.intrinsics[0, :] *= s_width
        self.intrinsics[1, :] *= s_height
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

    def get_intrinsics(self) -> np.ndarray:
        """return camera intrinsics"""
        return self.intrinsics

    def get_intrinsics_inv(self) -> np.ndarray:
        """return inverse of camera intrinsics"""
        return self.intrinsics_inv

    def get_projection(self) -> np.ndarray:
        """return 4x4 camera projection matrix"""
        # get camera data
        intrinsics = self.get_intrinsics()
        c2w = self.get_pose()
        # opencv standard
        intrinsics_padded = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
        w2c = np.linalg.inv(c2w)
        proj = intrinsics_padded @ w2c
        proj = np.concatenate([proj, np.zeros((1, 4))], axis=0)  # (4, 4)
        return proj

    def get_opengl_projection_matrix(
        self, near: float = 0.1, far: float = 100.0
    ) -> np.ndarray:
        # opengl standard
        projection_matrix = opengl_projection_matrix_from_intrinsics(
            self.get_intrinsics(), self.width, self.height, near, far
        )  # (4, 4)
        return projection_matrix

    def get_opengl_matrix_world(self) -> np.ndarray:
        w2c = self.get_pose_inv()
        matrix_world = opengl_matrix_world_from_w2c(w2c)
        return matrix_world

    def get_data_dict(self) -> dict:
        """return all camera data"""
        return self.data

    def has_rgbs(self) -> bool:
        """check if rgbs exists"""
        return self.data["rgbs"] is not None

    def get_rgbs(self) -> np.ndarray:
        """return, if exists, all camera frames"""
        if not self.has_rgbs():
            print_error("camera has no rgb frames")
        return self.data["rgbs"]

    def get_rgb(self, frame_idx: int = 0) -> np.ndarray:
        """returns, if exists, rgb at frame_idx"""
        if frame_idx >= self.temporal_dim:
            print_error("frame_idx out of bounds")
        return self.get_rgbs()[frame_idx]

    def has_masks(self) -> bool:
        """check if masks exists"""
        return self.data["masks"] is not None

    def get_masks(self) -> np.ndarray:
        """return, if exists, all camera masks, else None"""
        if not self.has_masks():
            print_error("camera has no mask frames")
        return self.data["masks"]

    def get_mask(self, frame_idx: int = 0) -> np.ndarray:
        """return, if exists, a mask at frame_idx, else None"""
        if frame_idx >= self.temporal_dim:
            print_error("frame_idx out of bounds")
        return self.get_masks()[frame_idx]

    def has_normals(self) -> bool:
        """check if normals exists"""
        return self.data["normals"] is not None

    def get_normals(self) -> np.ndarray:
        """return, if exists, all camera normal maps, else None"""
        if not self.has_normals():
            print_error("camera has no normal frames")
        return self.data["normals"]

    def get_normal(self, frame_idx: int = 0) -> np.ndarray:
        """return, if exists, the normal map at frame_idx, else None"""
        if frame_idx >= self.temporal_dim:
            print_error("frame_idx out of bounds")
        return self.get_normals()[frame_idx]

    def has_depths(self) -> bool:
        """check if depths exists"""
        return self.data["depths"] is not None

    def get_depths(self) -> np.ndarray:
        """return, if exists, all camera depth maps, else None"""
        if not self.has_depths():
            print_error("camera has no depth frames")
        return self.data["depths"]

    def get_depth(self, frame_idx: int = 0) -> np.ndarray:
        """return, if exists, the depth map at frame_idx, else None"""
        if frame_idx >= self.temporal_dim:
            print_error("frame_idx out of bounds")
        return self.get_depths()[frame_idx]

    def has_instance_masks(self) -> bool:
        """check if instance_masks exists"""
        return self.data["instance_masks"] is not None

    def get_instance_masks(self) -> np.ndarray:
        """return, if exists, all camera instance masks, else None"""
        if not self.has_instance_masks():
            print_error("camera has no instance mask frames")
        return self.data["instance_masks"]

    def get_instance_mask(self, frame_idx: int = 0) -> np.ndarray:
        """return, if exists, the instance mask at frame_idx, else None"""
        if frame_idx >= self.temporal_dim:
            print_error("frame_idx out of bounds")
        return self.get_instance_masks()[frame_idx]

    def has_semantic_masks(self) -> bool:
        """check if semantic_masks exists"""
        return self.data["semantic_masks"] is not None

    def get_semantic_masks(self) -> np.ndarray:
        """return, if exists, all camera semantic masks, else None"""
        if not self.has_semantic_masks():
            print_error("camera has no semantic mask frames")
        return self.data["semantic_masks"]

    def get_semantic_mask(self, frame_idx: int = 0) -> np.ndarray:
        """return, if exists, the semantic mask at frame_idx, else None"""
        if frame_idx >= self.temporal_dim:
            print_error("frame_idx out of bounds")
        return self.get_semantic_masks()[frame_idx]

    def get_pose(self) -> np.ndarray:
        """returns camera pose in world space"""
        pose = self.global_transform @ self.pose @ self.local_transform
        return pose

    def get_pose_inv(self) -> np.ndarray:
        """returns camera pose in world space"""
        pose = self.get_pose()
        pose_inv = np.linalg.inv(pose)
        return pose_inv

    def get_rotation(self) -> np.ndarray:
        """returns camera rotation in world space"""
        pose = self.get_pose()
        rotation = pose[:3, :3]
        return rotation

    def get_center(self) -> np.ndarray:
        """returns camera center in world space"""
        pose = self.get_pose()
        center = pose[:3, 3]
        return center

    def resize(self, subsample_factor) -> None:
        """make frames smaller by scaling them by scale factor (inplace operation)
        Args:
            subsample_factor (float): inverse of scale factor
            verbose (bool): print info
        """
        old_height, old_width = self.height, self.width
        scale = 1 / subsample_factor
        new_height = round(self.height * scale)
        new_width = round(self.width * scale)
        s_height = new_height / old_height
        s_width = new_width / old_width
        for modality_name in self.data.keys():
            self._subsample_modality(modality_name, scale=scale)
        # scale intrinsics accordingly
        self._scale_intrinsics(s_width, s_height)
        self.height, self.width = new_height, new_width
        if self.verbose:
            print(
                f"camera image plane resized from {old_height}, {old_width} to {self.height}, {self.width}"
            )

    def _subsample_modality(self, modality_name: str, scale: float) -> None:
        """subsample camera frames of given modality (inplace operation)"""

        if self.data[modality_name] is None:
            # skip
            return

        # subsample frames
        new_frames = []
        for frame in self.data[modality_name]:
            new_frame = cv.resize(
                frame, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA
            )
            if new_frame.ndim == 2:
                new_frame = new_frame[:, :, None]
            new_frames.append(new_frame)
        self.data[modality_name] = np.stack(new_frames)

    def get_rays(
        self,
        points_2d_screen: torch.Tensor = None,
        nr_rays_per_pixel: int = 1,
        jitter_pixels: bool = False,
        device: str = "cpu",
    ):
        """returns image rays origins and directions
        for 2d points on the image plane.
        If points are not provided, they are sampled
        from the image plane for every pixel.

        args:
            points_2d_screen (torch.Tensor, float or int, optional): (N, 2)
                                                Values in [0, W-1], [0, H-1].
                                                Default is None.
            device (str, optional): device to store tensors. Defaults to "cpu".
            jitter_pixels (bool, optional): Whether to jitter pixels.
                                            Only used if points_2d_screen is None.
                                            Defaults to False.
        out:
            rays_o (torch.Tensor): (N, 3)
            rays_d (torch.Tensor): (N, 3)
            points_2d_screen (torch.Tensor, float): (N, 2) screen space
                                            sampling coordinates

        """

        # sample points if not provided
        if points_2d_screen is None:

            assert nr_rays_per_pixel > 0, "nr_rays_per_pixel must be > 0"
            assert nr_rays_per_pixel == 1 or (
                nr_rays_per_pixel > 1 and jitter_pixels == True
            ), "jitter_pixels must be True if nr_rays_per_pixel > 1"

            pixels = get_pixels(self.height, self.width, device=device)  # (W, H, 2)
            # reshape pixels to (N, 2) repeat pixels nr_rays_per_pixel times
            pixels = pixels.reshape(-1, 2)  # (N, 2)
            pixels = pixels.repeat_interleave(nr_rays_per_pixel, dim=0)
            # get points in screen space
            points_2d_screen = get_points_2d_screen_from_pixels(
                pixels, jitter_pixels
            )  # (N, 2)

        c2w = torch.from_numpy(self.get_pose()).float().to(device)
        intrinsics_inv = torch.from_numpy(self.get_intrinsics_inv()).float().to(device)

        rays_o, rays_d = get_rays_per_points_2d_screen(
            c2w, intrinsics_inv, points_2d_screen
        )

        return rays_o, rays_d, points_2d_screen

    def get_data(
        self,
        points_2d_screen: torch.Tensor = None,
        frame_idx: int = 0,
        keys: list = ["rgbs", "masks"],
        device: str = "cpu",
    ):
        """return data values for points_2d_screen
        out:
            vals (dict):
                rgb_vals (optional, torch.Tensor): (N, 3)
                mask_vals (optional, torch.Tensor): (N, 1)
        """

        # sample points if not provided
        if points_2d_screen is None:
            pixels = get_pixels(self.height, self.width, device=device)  # (W, H, 2)
            points_2d_screen = get_points_2d_screen_from_pixels(pixels)  # (N, 2)

        # get data values
        frames_idx = torch.full(
            (points_2d_screen.shape[0],), frame_idx, dtype=torch.int32, device=device
        )
        
        data = {}
        for key in keys:
            if key not in self.data.keys() or self.data[key] is None:
                print_warning(f"data {key} not found for camera {self.camera_idx}")
            else:
                data[key] = self.data[key]
        
        vals = get_data_per_points_2d_screen(
            points_2d_screen=points_2d_screen,
            cameras_idx=None,
            frames_idx=frames_idx,
            data_dict=self.data,
        )

        # convert to torch tensors
        for key, val in vals.items():
            if val is not None:
                vals[key] = torch.from_numpy(val).to(device)

        return vals

    # TODO: deprecate
    # def get_random_rays_and_frames(
    #     self, nr_rays=512, frame_idx=0, jitter_pixels=False, device="cpu"
    # ):
    #     """given a camera and a number of rays, return random
    #     rays and images pixels values

    #     jitter_pixels (bool, optional): whether to jitter pixels. Defaults to False.
    #     """

    #     pixels = get_random_pixels(self.height, self.width, nr_rays, device=device)  # (N, 2)
    #     points_2d_screen = get_points_2d_screen_from_pixels(
    #         pixels, jitter_pixels
    #     )  # (N, 2)

    #     rays_o, rays_d, points_2d_screen = self.get_rays(
    #         points_2d_screen=points_2d_screen, device=device
    #     )

    #     return rays_o, rays_d, points_2d_screen

    def project_points_3d_world_to_2d_screen(
        self,
        points_3d: Union[np.ndarray, torch.Tensor],
        filter_points: bool = True,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Projects 3D points to 2D screen space using a camera object.

        Args:
            points_3d (np.ndarray or torch.Tensor): 3D points in world space of shape (N, 3).

        Returns:
            np.ndarray or torch.Tensor: 2D screen points of shape (N, 2).
            np.ndarray or torch.Tensor: Mask of points outside the image bounds.
        """

        # Retrieve camera intrinsics and pose
        intrinsics = self.get_intrinsics()
        c2w = self.get_pose()

        # Delegate to the helper function
        points_2d_screen = global_perspective_projection(intrinsics, c2w, points_3d)

        if filter_points:
            # Filter points outside the image bounds
            points_mask = get_mask_points_out_image(
                points_2d_screen, self.width, self.height
            )
            points_2d_screen = points_2d_screen[points_mask]
        else:
            points_mask = None

        return points_2d_screen, points_mask

    def unproject_points_2d_screen_to_3d_world(
        self,
        points_2d_screen: Union[np.array, torch.Tensor],
        depth: Union[np.array, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Unprojects 2D screen points to 3D world space using a camera object.

        Args:
            points_2d_screen (np.ndarray or torch.Tensor): 2D screen points of shape (N, 2).
            depth (np.ndarray or torch.Tensor): Depth values of shape (N,).

        Returns:
            np.ndarray or torch.Tensor: 3D points in world space of shape (N, 3).
        """

        # Validate input shapes
        if points_2d_screen.shape[0] != depth.shape[0]:
            raise ValueError(
                f"input shapes do not match: points_2d_screen: {points_2d_screen.shape} and depth: {depth.shape}."
            )
        if points_2d_screen.shape[1] != 2:
            raise ValueError(
                f"points_2d_screen: {points_2d_screen} must have shape (N, 2)."
            )
        if depth.ndim != 1:
            raise ValueError(f"depth: {depth.shape} must be a 1D array.")

        # Retrieve camera intrinsics and pose
        intrinsics_inv = self.get_intrinsics_inv()
        c2w = self.get_pose()

        # Delegate to the helper function
        points_3d_world = global_inv_perspective_projection(
            intrinsics_inv, c2w, points_2d_screen, depth
        )

        return points_3d_world

    def distance_to_points_3d_world(
        self, points_3d_world: Union[np.array, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the distance from the camera origin to 3D points in world space.

        Args:
            camera (Camera): Camera object with a `get_pose()` method.
            points_3d_world (np.ndarray or torch.Tensor): 3D points in world space of shape (N, 3).

        Returns:
            np.ndarray or torch.Tensor: Distances from the camera to the points, shape (N,).

        Raises:
            ValueError: If `points_3d_world` is of an unsupported type or shape.
        """

        # Retrieve camera pose
        c2w = self.get_pose()

        # Get camera center
        camera_center = c2w[:3, 3]

        # Compute the norm (distance to the camera)
        if isinstance(points_3d_world, torch.Tensor):
            camera_center = torch.tensor(
                camera_center, dtype=torch.float32, device=points_3d_world.device
            )
            points_3d_norm = torch.norm(points_3d_world - camera_center, dim=-1)
        elif isinstance(points_3d_world, np.ndarray):
            camera_center = np.asarray(camera_center, dtype=np.float32)
            points_3d_norm = np.linalg.norm(points_3d_world - camera_center, axis=-1)

        # TODO: compute distance from image plane to 3D points

        # # Convert or validate input types
        # if isinstance(points_3d_world, torch.Tensor):
        #     c2w = torch.tensor(c2w, dtype=torch.float32, device=points_3d_world.device)
        #     w2c = torch.inverse(c2w)  # World-to-camera transformation
        # elif isinstance(points_3d_world, np.ndarray):
        #     c2w = np.asarray(c2w, dtype=np.float32)
        #     w2c = np.linalg.inv(c2w)
        # else:
        #     raise ValueError("`points_3d_world` must be a torch.Tensor or np.ndarray.")

        # # Transform points from world space to camera space
        # points_3d_camera = apply_transformation_3d(points_3d_world, w2c)

        # # Compute the norm (distance to the camera)
        # if isinstance(points_3d_world, torch.Tensor):
        #     points_3d_norm = torch.norm(points_3d_camera, dim=-1)
        # elif isinstance(points_3d_world, np.ndarray):
        #     points_3d_norm = np.linalg.norm(points_3d_camera, axis=-1)
        # else:
        #     raise ValueError("`points_3d` must be a torch.Tensor or np.ndarray.")

        return points_3d_norm

    def __str__(self) -> str:
        """print camera information"""
        string = f"camera_idx: {self.camera_idx}\n"
        string += f"intrinsics ({self.intrinsics.dtype}):\n"
        string += str(self.intrinsics) + "\n"
        string += f"pose ({self.pose.dtype}):\n"
        string += str(self.pose) + "\n"
        string += f"global_transform ({self.global_transform.dtype}):\n"
        string += str(self.global_transform) + "\n"
        string += f"local_transform ({self.local_transform.dtype}):\n"
        string += str(self.local_transform) + "\n"
        for modality_name, modality_frames in self.data.items():
            if modality_frames is None:
                # skip empty data
                continue
            string += modality_name + f" ({modality_frames.dtype}):\n"
            string += str(modality_frames.shape) + "\n"

        return string
