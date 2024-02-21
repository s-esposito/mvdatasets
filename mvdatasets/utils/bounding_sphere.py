import torch
import numpy as np

# from mvdatasets.utils.geometry import apply_transformation_3d
from mvdatasets.utils.geometry import deg2rad


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
    
    if rays_o.device != center.device:
        raise ValueError("rays and bounding sphere must be on the same device")
    
    # general case: camera eye outside sphere
    oc = rays_o - center
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
    is_inside = torch.norm(rays_o - center, dim=1) < radius
    is_hit[is_inside] = True
    t_near[is_inside] = 0.0
    
    t_near[~is_hit] = 0.0
    t_far[~is_hit] = 0.0
    
    # oc = rays_o - center
    # a = torch.sum(rays_d * rays_d, dim=1)
    # b = 2.0 * torch.sum(oc * rays_d, dim=1)
    # c = torch.sum(oc * oc, dim=1) - radius * radius

    # # Compute half-discriminant
    # half_discriminant = b * b / (4.0 * a)

    # # Compute intersection points
    # t_near = (-b - torch.sqrt(half_discriminant)) / (2.0 * a)
    # t_far = (-b + torch.sqrt(half_discriminant)) / (2.0 * a)

    # # Early exit if no intersection
    # is_hit = (t_near <= t_far) & (half_discriminant >= 0)

    # # Handle camera eye inside sphere
    # is_inside = torch.norm(rays_o - center, dim=1) < radius
    # is_hit[is_inside] = True
    # t_near[is_inside] = 0.0

    # # Set negative t_near to zero
    # t_near[t_near < 0] = 0.0

    # # Set t_far to zero for non-hits
    # t_far[~is_hit] = 0.0

    return is_hit, t_near, t_far


class BoundingSphere:
    
    def __init__(
            self,
            pose=np.eye(4),
            local_scale=np.array([1, 1, 1]),
            label=None,
            color=None,
            line_width=1.0,
            device="cpu"
        ):
        """Bounding Sphere class.

        Args:
            pose (np.ndarray, optional): Defaults to np.eye(4).
            local_scale (np.ndarray, optional): Defaults to np.array([1, 1, 1]).
            label (str, optional): Defaults to None.
            color (str, optional): Defaults to None.
            line_width (float, optional): Defaults to 1.0.
            device (str, optional): Defaults to "cpu".
        """
        
        assert local_scale.shape == (3,)
        assert local_scale[0] == local_scale[1] == local_scale[2], "only isotropic scaling is currently supported for spheres"
        
        self.pose = torch.tensor(pose, dtype=torch.float32, device=device)
        self.local_scale = torch.tensor(local_scale, dtype=torch.float32, device=device)
        self.device = device
        
        # mostly useful for visualization
        self.label = label
        self.color = color  # matplotlib color
        self.line_width = line_width
        
    def get_pose(self):
        return self.pose
    
    def get_center(self):
        pose = self.get_pose()
        return pose[:3, 3]
    
    def get_radius(self):
        return self.local_scale[0].item()
    
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
        
        # pose in world space
        pose = self.get_pose()
        
        if self.pose.device != rays_o.device:
            raise ValueError("rays and bounding sphere must be on the same device")
        
        center = pose[:3, 3]
        radius = self.local_scale[0]
        
        # rays are already in world space
        
        is_hit, t_near, t_far = _intersect_sphere(
            rays_o, rays_d, center, radius
        )
        p_near = rays_o + rays_d * t_near[:, None]
        p_far = rays_o + rays_d * t_far[:, None]
        
        return is_hit, t_near, t_far, p_near, p_far
    
    def get_random_points_inside(self, nr_points):
        # points in local space
        eps = 1e-6
        azimuth_deg = torch.rand(nr_points, device=self.device) * 360
        elevation_deg = torch.rand(nr_points, device=self.device) * 180 - 90
        radius = torch.rand(nr_points, device=self.device) * (self.get_radius() - eps)
        azimuth_rad = deg2rad(azimuth_deg)
        elevation_rad = deg2rad(elevation_deg)
        x = torch.cos(azimuth_rad) * torch.cos(elevation_rad) * radius
        y = torch.sin(elevation_rad) * radius  # y is up
        z = torch.sin(azimuth_rad) * torch.cos(elevation_rad) * radius
        points = torch.column_stack((x, y, z))
        return points
    
    def check_points_inside(self, points):
        points_ = points - self.get_center()
        # get l2 norm of points
        points_norm = torch.norm(points_, dim=1)
        return (points_norm < self.get_radius())
    
    def save_as_ply():
        print("WARNING: saving as ply is not currently supported for BoundingSphere")