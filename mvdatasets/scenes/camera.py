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
    """Camera class.
    
    Assumptions:
    - all modalities have the same dimensions
    """

    def __init__(
        self, intrinsics, pose, 
        rgbs=None, masks=None, normals=None, depths=None,
        instance_masks=None, semantic_masks=None,
        global_transform=None, local_transform=None,
        camera_idx=0
    ):
        """Create a camera object, all parameters are np.ndarrays.

        Args:
            rgbs (np.array, float): (T, H, W, 3) with values in [0, 1]
            masks (np.array, float): (T, H, W, 1) with values in [0, 1]
            normals (np.array, float): (T, H, W, 3) with values in [0, 1]
            depths (np.array, float): (T, H, W, 1) with values in [0, 1]
            instance_masks (np.array, int): (T, H, W, 1) with values in [0, n_instances]
            semantic_masks (np.array, int): (T, H, W, 1) with values in [0, n_classes]
            intrinsics (np.array): (3, 3)
            pose (np.array): (4, 4)
        """

        # assert shapes are correct
        assert intrinsics.shape == (3, 3)
        assert pose.shape == (4, 4)
        if rgbs is not None:
            assert rgbs.ndim == 4 and rgbs.shape[-1] == 3
        if masks is not None:
            assert masks.ndim == 4 and masks.shape[-1] == 1
        
        self.camera_idx = camera_idx
        self.intrinsics = intrinsics
        self.pose = pose
        self.intrinsics_inv = np.linalg.inv(intrinsics)
        
        # load modalities
        self.modalities = {}
        
        # rgbs
        if rgbs is not None:
            self.modalities["rgb"] = rgbs
            
        # masks
        if masks is not None:
            self.modalities["mask"] = masks
            
        # normals
        if normals is not None:
            self.modalities["normal"] = normals
        
        # depths
        if depths is not None:
            self.modalities["depth"] = depths
        
        # instance_masks
        if instance_masks is not None:
            self.modalities["instance_mask"] = instance_masks
        
        # semantic_masks
        if semantic_masks is not None:
            self.modalities["semantic_mask"] = semantic_masks
            
        # frames dims
        self.height, self.width = self.get_screen_space_dims()

        # transforms
        if global_transform is not None:
            self.global_transform = global_transform
        else:
            self.global_transform = np.eye(4)
            
        if local_transform is not None:
            self.local_transform = local_transform
        else:
            self.local_transform = np.eye(4)

    def has_modality(self, modality_name):
        """check if modality_name exists in frames"""
        return modality_name in self.modalities
    
    def get_modality_dims(self, modality_name):
        """return frame dims
        
        out:
            (height, width)
        """
        return self.modalities[modality_name].shape[1:3]
    
    def get_screen_space_dims(self):
        """return image plane dims (height, width)"""
        modality_names = list(self.modalities.keys())
        if len(modality_names) == 0:
            # camera has no images attached
            return 0, 0
        
        # get dims of first modality
        modality_name = modality_names[0]
        height, width = self.get_modality_dims(modality_name)
        
        return height, width
    
    def get_modality_frames(self, modality_name):
        """return all camera frames of type modality_name"""
        if modality_name not in self.modalities:
            return None
        return self.modalities[modality_name]
    
    def get_modality_frame(self, modality_name, frame_idx=0):
        """return frame at frame_idx"""
        if modality_name not in self.modalities:
            return None
        if len(self.modalities[modality_name]) <= frame_idx:
            return None
        frame = self.modalities[modality_name][frame_idx]
        
        # unsqueeze if frame is 2D
        if frame.ndim == 2:
            frame = frame[:, :, None]

        return frame
    
    def resize(self, max_dim, verbose=False):
        """make frames smaller by iteratively scaling dimensions by 0.5
        until we reach a certain size.

        Args:
            modality_name (str): modality name
            max_dim (int): target max dimension (width or height)
        """
        
        scale_factor = 1
        old_height, old_width = self.height, self.width
        while min(self.width, self.height) > max_dim:
            for modality_name in self.modalities.keys():
                self.subsample_modality(modality_name, scale=0.5)
            scale_factor *= 0.5
            self.height, self.width = self.get_screen_space_dims()
        # scale intrinsics accordingly
        self.scale_intrinsics(scale_factor)
        if verbose:
            print(f"camera image plane resized from {old_height}, {old_width} to {self.height}, {self.width}")
                    
    def scale_intrinsics(self, scale):
        """scales the intrinsics matrix"""
        self.intrinsics[0, 0] *= scale
        self.intrinsics[1, 1] *= scale
        self.intrinsics[0, 2] *= scale
        self.intrinsics[1, 2] *= scale
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

    def get_intrinsics(self):
        """return camera intrinsics"""
        return self.intrinsics

    def get_intrinsics_inv(self):
        """return inverse of camera intrinsics"""
        return self.intrinsics_inv

    def has_rgbs(self):
        """check if rgbs exists"""
        return self.has_modality("rgb")
    
    def get_rgbs(self):
        """return, if exists, all camera frames"""
        return self.get_modality_frames("rgb")

    def get_rgb(self, frame_idx=0):
        """returns, if exists, image at frame_idx"""
        return self.get_modality_frame("rgb", frame_idx=frame_idx)

    def has_masks(self):
        """check if masks exists"""
        return self.has_modality("mask")
    
    def get_masks(self):
        """return, if exists, all camera masks, else None"""
        return self.get_modality_frames("mask")

    def get_mask(self, frame_idx=0):
        """return, if exists, a mask at frame_idx, else None"""
        return self.get_modality_frame("mask", frame_idx=frame_idx)
    
    def has_normals(self):
        """check if normals exists"""
        return self.has_modality("normal")
    
    def get_normals(self):
        """return, if exists, all camera normal maps, else None"""
        return self.get_modality_frames("normal")
    
    def get_normal(self, frame_idx=0):
        """return, if exists, the normal map at frame_idx, else None"""
        return self.get_modality_frame("normal", frame_idx=frame_idx)
    
    def has_depths(self):
        """check if depths exists"""
        return self.has_modality("depth")
    
    def get_depths(self):
        """return, if exists, all camera depth maps, else None"""
        return self.get_modality_frames("depth")
    
    def get_depth(self, frame_idx=0):
        """return, if exists, the depth map at frame_idx, else None"""
        return self.get_modality_frame("depth", frame_idx=frame_idx)
    
    def has_instance_masks(self):
        """check if instance_masks exists"""
        return self.has_modality("instance_mask")
    
    def get_instance_masks(self):
        """return, if exists, all camera instance masks, else None"""
        return self.get_modality_frames("instance_mask")
    
    def get_instance_mask(self, frame_idx=0):
        """return, if exists, the instance mask at frame_idx, else None"""
        return self.get_modality_frame("instance_mask", frame_idx=frame_idx)
    
    def has_semantic_masks(self):
        """check if semantic_masks exists"""
        return self.has_modality("semantic_mask")
    
    def get_semantic_masks(self):
        """return, if exists, all camera semantic masks, else None"""
        return self.get_modality_frames("semantic_mask")
    
    def get_semantic_mask(self, frame_idx=0):
        """return, if exists, the semantic mask at frame_idx, else None"""
        return self.get_modality_frame("semantic_mask", frame_idx=frame_idx)

    def get_pose(self):
        """returns camera pose in world space"""
        pose = self.global_transform @ self.pose @ self.local_transform
        return pose

    def concat_global_transform(self, global_transform):
        # apply global_transform
        self.global_transform = global_transform @ self.global_transform

    def subsample_modality(self, modality_name, scale):
        """subsample camera frames of given modality (inplace operation)"""
        
        # subsample frames
        new_frames = []
        for frame in self.get_modality_frames(modality_name):
            new_frames.append(
                cv.resize(
                            frame,
                            (0, 0),
                            fx=scale,
                            fy=scale,
                            interpolation=cv.INTER_AREA
                        )
            )
        self.modalities[modality_name] = np.stack(new_frames)

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
        for modality_name, modality_frames in self.modalities.items():
            string += modality_name + ":\n"
            string += str(modality_frames.shape) + "\n"

        return string
