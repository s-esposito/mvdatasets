import torch


class Scene:
    def __init__(self, cameras) -> None:
        """Scene constructor

        Args:
            cameras (Camera[]): list of cameras
        """
        self.cameras = cameras
