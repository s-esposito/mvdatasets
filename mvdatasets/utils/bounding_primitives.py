import numpy as np

"""collection of scene bounding primitives"""


class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect(self, rays):
        is_hit = np.zeros(rays.shape[0], dtype=np.bool)
        t_near = np.zeros(rays.shape[0], dtype=np.float32)
        t_far = np.ones(rays.shape[0], dtype=np.float32)

        # TODO: implement ray-sphere intersection
        
        return is_hit, t_near, t_far


class AABB:
    def __init__(self, min_vert, max_vert):
        self.min_vert = min_vert
        self.max_vert = max_vert

    def intersect(self, rays):
        is_hit = np.zeros(rays.shape[0], dtype=np.bool)
        t_near = np.zeros(rays.shape[0], dtype=np.float32)
        t_far = np.ones(rays.shape[0], dtype=np.float32)

        # TODO: implement ray-AABB intersection
        
        return is_hit, t_near, t_far
