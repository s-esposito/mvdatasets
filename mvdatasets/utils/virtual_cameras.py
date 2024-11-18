import numpy as np
from typing import List
from mvdatasets import Camera
from mvdatasets.geometry.common import look_at, deg2rad


def sample_cameras_on_hemisphere(
    intrinsics: np.ndarray,
    width: int,
    height: int,
    radius: float = 1.0,
    nr_cameras: int = 10,
    up: np.ndarray = np.array([0, 1, 0]),
    center: np.ndarray = np.array([0, 0, 0]),
) -> List[Camera]:

    azimuth_deg = np.random.uniform(0, 360, nr_cameras)
    elevation_deg = np.random.uniform(-90, 90, nr_cameras)
    azimuth_rad = deg2rad(azimuth_deg)
    elevation_rad = deg2rad(elevation_deg)
    x = np.cos(azimuth_rad) * np.cos(elevation_rad) * radius
    y = np.sin(elevation_rad) * radius  # y is up
    z = np.sin(azimuth_rad) * np.cos(elevation_rad) * radius
    # x = np.array(x)
    # y = np.array(y)
    # z = np.array(z)
    cameras_centers = np.column_stack((x, y, z))
    
    cameras = []
    for i in range(nr_cameras):

        # get rotation matrix from azimuth and elevation
        pose = np.eye(4)
        pose = look_at(cameras_centers[i], center, up)

        # local transform
        local_transform = np.eye(4)
        local_transform[:3, :3] = np.array(
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32
        )

        camera = Camera(
            intrinsics,
            pose,
            width=width,
            height=height,
            local_transform=local_transform,
            camera_idx=i,
        )
        cameras.append(camera)

    return cameras
