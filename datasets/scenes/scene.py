# import torch


class Scene:
    def __init__(self, cameras, point_cloud=None):
        """Scene constructor

        Args:
            cameras (Camera[]): list of cameras
            point_cloud (np.array): (N, 3) point cloud (optional)
        """
        self.cameras = cameras
        self.point_cloud = point_cloud
