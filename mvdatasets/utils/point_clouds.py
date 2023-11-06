import open3d as o3d
import os
import numpy as np


def load_point_clouds(point_clouds_paths, max_nr_points=1000):
    """Loads point cloud from files.
    If the file is a mesh, points are its vertices.

    Args:
        point_clouds_paths: ordered list of point cloud file paths
        max_nr_points: maximum number of points to load

    Returns:
        point_clouds []: ordered list of (N, 3) numpy arrays
    """

    point_clouds = []
    for pc_path in point_clouds_paths:
        # if exists, load it
        if os.path.exists(pc_path):
            # if format is .ply or .obj
            if pc_path.endswith(".ply") or pc_path.endswith(".obj"):
                print("Loading point cloud from {}".format(pc_path))
                point_cloud = o3d.io.read_point_cloud(pc_path)
                points_3d = np.asarray(point_cloud.points)
                if points_3d.shape[0] > max_nr_points:
                    # downsample
                    random_idx = np.random.choice(
                        points_3d.shape[0], max_nr_points, replace=False
                    )
                    points_3d = points_3d[random_idx]
                    point_clouds.append(points_3d)
                print("Loaded {} points from mesh".format(points_3d.shape[0]))
            else:
                raise ValueError("Unsupported point cloud format")
        else:
            raise ValueError("Point cloud path {} does not exist".format(pc_path))

    return point_clouds
