import torch
import numpy as np

# from mvdatasets.utils.geometry import apply_transformation_3d


def _intersect_sphere(rays_o, rays_d, center, radius):
    """
    Args:
        rays_o (torch.tensor): (N, 3)
        rays_d (torch.tensor): (N, 3)
        center (torch.tensor): (3,)
        radius (float)
    Out: 
        is_hit (torch.tensor): (N,)
        t_near (torch.tensor): (N,)
        t_far (torch.tensor): (N,)
    """
    
    center_ = torch.tensor(center, dtype=rays_o.dtype, device=rays_o.device)
    
    # general case: camera eye outside sphere
    oc = rays_o - center_
    a = torch.sum(rays_d * rays_d, dim=1)
    b = 2.0 * torch.sum(oc * rays_d, dim=1)
    c = torch.sum(oc * oc, dim=1) - radius*radius
    discriminants = b*b - 4*a*c
    invalid_discriminats = (discriminants < 0)

    discriminants[invalid_discriminats] = 0
    sqrt_discriminants = torch.sqrt(discriminants)
    t_near = (-b - sqrt_discriminants) / (2.0*a)
    t_far = (-b + sqrt_discriminants) / (2.0*a)
    
    t_near[t_near < 0] = 0.0
    is_hit = (t_near <= t_far) & (~invalid_discriminats)
    
    # special case: camera eye inside sphere
    # check if rays_o distance from center is less than radius
    is_inside = torch.norm(rays_o - center_, dim=1) < radius
    is_hit[is_inside] = True
    t_near[is_inside] = 0.0
    
    t_near[~is_hit] = 0.0
    t_far[~is_hit] = 0.0

    return is_hit, t_near, t_far


class BoundingSphere:
    
    def __init__(
            self,
            pose=np.eye(4),
            local_scale=np.array([1, 1, 1]),
            label=None,
            color=None,
            line_width=1.0,
        ):
        
        assert local_scale.shape == (3,)
        assert local_scale[0] == local_scale[1] == local_scale[2], "only isotropic scaling is currently supported for spheres"
        
        self.pose = pose
        self.local_scale = local_scale
        
        # mostly useful for visualization
        self.label = label
        self.color = color  # matplotlib color
        self.line_width = line_width
        
    def get_pose(self):
        return self.pose
    
    def intersect(self, rays_o, rays_d):
        """
        Args:
            rays in world space
            rays_o (torch.tensor): (N, 3)
            rays_d (torch.tensor): (N, 3)
        """
        
        # pose in world space
        pose = self.get_pose()
        center = pose[:3, 3]
        radius = self.local_scale[0]
        
        # rays are already in world space
        
        is_hit, t_near, t_far = _intersect_sphere(
            rays_o, rays_d, center, radius
        )
        p_near = rays_o + rays_d * t_near[:, None]
        p_far = rays_o + rays_d * t_far[:, None]
        
        return is_hit, t_near, t_far, p_near, p_far
    
    def save_as_ply():
        print("WARNING: saving as ply is not currently supported for BoundingSphere")