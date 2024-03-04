from rich import print
import torch
import os
import numpy as np
from copy import deepcopy
import open3d as o3d

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
    
    if rays_o.device != aabb_min.device:
        raise ValueError("rays and bounding box must be on the same device")
    
    # avoid numeric issues
    eps = 1e-6
    aabb_min = aabb_min + eps
    aabb_max = aabb_max - eps
    
    # general case: camera eye outside bb
    t_min = (aabb_min - rays_o) / rays_d
    t_max = (aabb_max - rays_o) / rays_d
    t1 = torch.min(t_min, t_max)
    t2 = torch.max(t_min, t_max)
    t_near = torch.max(torch.max(t1[:, 0], t1[:, 1]), t1[:, 2])
    t_far = torch.min(torch.min(t2[:, 0], t2[:, 1]), t2[:, 2])
    
    t_near[t_near < 0] = 0.0
    is_hit = t_near <= t_far
    
    # special case: camera eye inside bb
    is_inside = (rays_o >= aabb_min).all(dim=1) & (rays_o <= aabb_max).all(dim=1)
    is_hit[is_inside] = True
    t_near[is_inside] = 0.0
    
    t_near[~is_hit] = 0.0
    t_far[~is_hit] = 0.0
    
    return is_hit, t_near, t_far


class BoundingBox:
    
    def __init__(
            self,
            pose=np.eye(4),
            local_scale=np.array([1, 1, 1]),
            father_bb=None,
            label=None,
            color=None,
            line_width=1.0,
            device="cpu",
            verbose=True
        ):
        """Bounding Box class.

        Args:
            pose (np.ndarray, optional): Defaults to np.eye(4).
            local_scale (np.ndarray, optional): Defaults to np.array([1, 1, 1]).
            father_bb (BoundingBox, optional): Defaults to None.
            label (str, optional): Defaults to None.
            color (str, optional): Defaults to None.
            line_width (float, optional): Defaults to 1.0.
            device (str, optional): Defaults to "cpu".
        """
        
        assert local_scale.shape == (3,)
        
        # pose in father bounding box space
        # or world space if father_bb is None
        self.pose = torch.tensor(pose, dtype=torch.float32, device=device)
        
        # reference to father bounding box
        self.father_bb = father_bb
        if father_bb is not None:
            if self.father_bb.device != device:
                raise ValueError("father and child and bounding boxes must be on the same device")
        
        # if self.father_bb is None:
        self.local_scale = torch.tensor(local_scale, dtype=torch.float32, device=device)
        # else:
        # inherit local scale from father bounding box
        # self.local_scale = self.father_bb.local_scale
            
        self.device = device
        self.identity = torch.eye(4, dtype=torch.float32, device=device)
        
        # mostly useful for visualization
        self.label = label
        self.color = color  # matplotlib color
        self.line_width = line_width
        
        if verbose:
            print(f"created bounding box with local sides lenghts: {self.local_scale.cpu().numpy()}")
    
    def get_pose(self):
        pose = deepcopy(self.pose)
        father_bb = self.father_bb
        while father_bb is not None:
            pose = father_bb.get_pose() @ pose
            father_bb = father_bb.father_bb
        return pose
    
    def get_radius(self):
        return (torch.max(self.local_scale) / 2.0).item()
    
    def get_center(self):
        pose = self.get_pose()
        return pose[:3, 3]
    
    def get_vertices(self, in_world_space=False):
        
        # offsets
        center = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        offsets = torch.tensor([
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1]
        ], dtype=torch.float32, device=self.device)
        
        # vertices in bounding box space
        vertices = center + offsets * self.local_scale / 2
        
        # conver to world space
        pose = self.get_pose()
        if in_world_space:
            vertices = apply_transformation_3d(vertices, pose)
        
        return vertices
    
    def intersect(self, rays_o, rays_d):
        """
        Args:
            rays in world space
            rays_o (torch.tensor): (N, 3)
            rays_d (torch.tensor): (N, 3)
        Out:
            is_hit (torch.tensor): (N,)
            t_near (torch.tensor): (N,)
            t_far (torch.tensor): (N,)
            p_near (torch.tensor): (N, 3)
            p_far (torch.tensor): (N, 3)
        """
        
        pose = self.get_pose()
        
        if self.pose.device != rays_o.device:
            raise ValueError("rays and bounding box must be on the same device")
        
        if (pose != self.identity).any():
            # convert rays to bounding box space
            inv_pose = torch.linalg.inv(pose)
            rays_o = apply_transformation_3d(rays_o, inv_pose)
            rays_d = (inv_pose[:3, :3] @ rays_d.T).T
            
        # compute intersections in bounding box space
        aabb_min = -1 * self.local_scale / 2
        aabb_max = self.local_scale / 2
        is_hit, t_near, t_far = _intersect_aabb(
            rays_o, rays_d, aabb_min, aabb_max
        )
        p_near = rays_o + rays_d * t_near[:, None]
        p_far = rays_o + rays_d * t_far[:, None]
        
        if (pose != self.identity).any():
            # convert intersections to world space
            p_near = apply_transformation_3d(p_near, pose)
            p_far = apply_transformation_3d(p_far, pose)
        
        return is_hit, t_near, t_far, p_near, p_far
    
    @torch.no_grad()
    def get_random_points_inside(self, nr_points, in_world_space=False, padding=0.0):
        # points in local space
        eps = 1e-6 + padding
        scale = (self.local_scale / 2) - eps
        points = torch.rand((nr_points, 3), dtype=torch.float32, device=self.device) * 2.0 - 1.0
        points = points * scale
        
        if in_world_space:
            # convert to world space
            pose = self.get_pose()
            points = apply_transformation_3d(points, pose)
            
        return points
    
    @torch.no_grad()
    def get_random_points_on_surface(self, nr_points, in_world_space=False):
        # points in local space
        
        # approximate nr_points to closest multiple of 12
        nr_points_per_face = nr_points // 12
        nr_points_ = nr_points_per_face * 12
        
        # bbox vertices
        vertices = self.get_vertices()  # vertices in local space

        # define faces of the box
        faces = torch.tensor([
            [0, 4, 5],
            [0, 5, 1],
            [4, 6, 7],
            [4, 7, 5],
            [2, 0, 1],
            [2, 1, 3],
            [2, 3, 7],
            [2, 7, 6],
            [1, 5, 7],
            [1, 7, 3],
            [2, 6, 4],
            [2, 4, 0]
        ], dtype=torch.int32, device=self.device)
        
        tri_vertices = vertices[faces.view(-1)]
        tri_vertices = tri_vertices.reshape(-1, 3, 3)
        
        bar_coords = torch.rand((nr_points_, 3), dtype=torch.float32, device=self.device)
        bar_coords = bar_coords / bar_coords.sum(dim=1, keepdim=True)
        bar_coords = bar_coords.reshape(-1, nr_points_per_face, 3, 1)
        
        bar_coords_reshaped = bar_coords  # (12, nr_points_per_face, 3, 1)
        tri_vertices_reshaped = tri_vertices.unsqueeze(1)  # (12, 1, 3, 3)
        multipled_tensors = tri_vertices_reshaped * bar_coords_reshaped  # (12, nr_points_per_face, 3, 3)
        points = torch.sum(multipled_tensors, dim=-2) # (12, nr_points_per_face, 3)
        points = points.reshape(-1, 3)
        
        if in_world_space:
            # convert to world space
            pose = self.get_pose()
            points = apply_transformation_3d(points, pose)
        
        return points
    
    @torch.no_grad()
    def check_points_inside(self, points):
        """checks if points are inside the bounding box

        args:
            points (torch.tensor): (N, 3) points are in world space

        returns:
            is_inside (torch.tensor): (N,)
        """
        
        pose = self.get_pose()
        
        if (pose != self.identity).any():
            # convert rays to bounding box space
            inv_pose = torch.linalg.inv(pose)
            points_ = apply_transformation_3d(points, inv_pose)
        else:
            points_ = points
        
        vertices = self.get_vertices()  # vertices in local space
        vmin = vertices[0]
        vmax = vertices[7]
        
        cond_1 = (points_ > vmin).all(dim=1)
        cond_2 = (points_ < vmax).all(dim=1)
        is_inside = torch.logical_and(cond_1, cond_2)
        
        return is_inside
    
    def save_as_ply(self, dir_path, name):
        
        # bbox vertices
        vertices = self.get_vertices(in_world_space=True)  # vertices in world space
        vertices = vertices.cpu().numpy()
        
        # define faces of the box
        faces = np.array([
            [0, 4, 5],
            [0, 5, 1],
            [4, 6, 7],
            [4, 7, 5],
            [2, 0, 1],
            [2, 1, 3],
            [2, 3, 7],
            [2, 7, 6],
            [1, 5, 7],
            [1, 7, 3],
            [2, 6, 4],
            [2, 4, 0]
        ])
        
        # create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # save mesh path
        o3d.io.write_triangle_mesh(os.path.join(dir_path, f"{name}.ply"), mesh)
        
        return mesh