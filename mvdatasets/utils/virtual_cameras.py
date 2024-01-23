import numpy as np

from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.geometry import look_at, deg2rad


def sample_cameras_on_hemisphere(
        intrinsics, width, height,
        radius=1.0, nr_cameras=10,
        up=np.array([0, 1, 0]),
        center=np.array([0, 0, 0])
    ):
    
    # # azimuth_deg = np.linspace(0, 360, nr_cameras, endpoint=False)
    # # elevation_deg = np.linspace(0, 45, nr_cameras, endpoint=False)
    azimuth_deg = np.random.uniform(0, 360, nr_cameras)
    elevation_deg = np.random.uniform(-90, 90, nr_cameras)
    # print("ele", elevation_deg)
    azimuth_rad = deg2rad(azimuth_deg)
    elevation_rad = deg2rad(elevation_deg)
    x = np.cos(azimuth_rad) * np.cos(elevation_rad) * radius
    y = np.sin(elevation_rad) * radius # y is up
    z = np.sin(azimuth_rad) * np.cos(elevation_rad) * radius
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    cameras_centers = np.column_stack((x, y, z))
    
    # points = np.random.uniform(-1, 1, (nr_cameras, 3))
    # points = points / np.linalg.norm(points, axis=1, keepdims=True)
    # cameras_centers = points * radius
    
    cameras = []
    for i in range(nr_cameras):

        # get rotation matrix from azimuth and elevation
        pose = np.eye(4)
        rotation = look_at(cameras_centers[i], center, up)
        pose[:3, :3] = rotation
        pose[:3, 3] = cameras_centers[i]
        
        # local transform
        local_transform = np.eye(4)
        local_transform[:3, :3] = np.array([[-1, 0, 0],
                                            [0, -1, 0],
                                            [0, 0, -1]], dtype=np.float32)
    
        camera = Camera(
                            intrinsics,
                            pose,
                            width=width,
                            height=height,
                            local_transform=local_transform,
                            camera_idx=i
                        )
        cameras.append(camera)
    
    return cameras