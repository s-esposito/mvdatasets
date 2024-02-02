import torch
import numpy as np
from copy import deepcopy

from mvdatasets.utils.geometry import apply_transformation_3d

def _intersect_aabb(rays_o, rays_d, aabb_min, aabb_max):
    """
    Args:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
        aabb_min (np.ndarray): (3,)
        aabb_max (np.ndarray): (3,)
    Out:
        is_hit (torch.tensor): (N,)
        t_near (torch.tensor): (N,)
        t_far (torch.tensor): (N,)
    """
    
    vmin = torch.tensor(aabb_min, dtype=rays_o.dtype, device=rays_o.device)
    vmax = torch.tensor(aabb_max, dtype=rays_o.dtype, device=rays_o.device)
    
    t_min = (vmin - rays_o) / rays_d
    t_max = (vmax - rays_o) / rays_d
    t1 = torch.min(t_min, t_max)
    t2 = torch.max(t_min, t_max)
    t_near = torch.max(torch.max(t1[:, 0], t1[:, 1]), t1[:, 2])
    t_far = torch.min(torch.min(t2[:, 0], t2[:, 1]), t2[:, 2])
    
    is_hit = t_far > 0.0
    is_hit = is_hit * (t_near < t_far)
    
    t_near[~is_hit] = 0.0
    t_far[~is_hit] = 0.0
    
    return is_hit, t_near, t_far


class BoundingBox:
    
    def __init__(
            self,
            pose=np.eye(4),
            sizes=np.array([1, 1, 1]),
            father_bb=None,
            label=None,
            color=None,
            line_width=1.0,
        ):
        """Bounding box class.

        Args:
            pose (np.ndarray, optional): Defaults to np.eye(4).
            sizes (np.ndarray, optional): Defaults to np.array([1, 1, 1]).
            father_bb (BoundingBox, optional): Defaults to None.
            label (str, optional): Defaults to None.
            color (str, optional): Defaults to None.
            line_width (float, optional): Defaults to 1.0.
        """
        
        # pose in father bounding box space
        # or world space if father_bb is None
        self.pose = pose
        # reference to father bounding box
        self.father_bb = father_bb
        if self.father_bb is None:
            assert sizes is not None
            self.sizes = sizes
        else:
            self.sizes = self.father_bb.sizes
        
        # mostly useful for visualization
        self.label = label
        self.color = color  # matplotlib color
        self.line_width = line_width
    
    def get_pose(self):
        pose = deepcopy(self.pose)
        father_bb = self.father_bb
        while father_bb is not None:
            pose = father_bb.get_pose() @ pose
            father_bb = father_bb.father_bb
        return pose
    
    def get_vertices(self):
        
        # offsets
        center = np.array([0, 0, 0])
        offsets = np.array([
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1]
        ])
        
        # vertices in bounding box space
        vertices = center + offsets * self.sizes / 2
        
        # conver to world space
        pose = self.get_pose()
        vertices = apply_transformation_3d(vertices, pose)
        
        return vertices
    
    def intersect(self, rays_o, rays_d):
        """
        Args:
            rays in world space
            rays_o (torch.tensor): (N, 3)
            rays_d (torch.tensor): (N, 3)
        """
        
        # convert rays to bounding box space
        pose = self.get_pose()
        pose = torch.tensor(pose, dtype=rays_o.dtype, device=rays_o.device)
        inv_pose = torch.linalg.inv(pose)
                
        rays_o = apply_transformation_3d(rays_o, inv_pose)
        rays_d = (inv_pose[:3, :3] @ rays_d.T).T
        
        # compute intersections in bounding box space
        aabb_min = np.array([-1, -1, -1]) * self.sizes / 2
        aabb_max = np.array([1, 1, 1]) * self.sizes / 2
        is_hit, t_near, t_far = _intersect_aabb(
            rays_o, rays_d, aabb_min, aabb_max
        )
        p_near = rays_o + rays_d * t_near[:, None]
        p_far = rays_o + rays_d * t_far[:, None]
        
        # convert intersections to world space
        p_near = apply_transformation_3d(p_near, pose)
        p_far = apply_transformation_3d(p_far, pose)
        
        return is_hit, t_near, t_far, p_near, p_far