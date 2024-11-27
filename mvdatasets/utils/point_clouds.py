from rich import print
import open3d as o3d
import os
import numpy as np
from mvdatasets.utils.printing import print_error, print_warning
from mvdatasets.geometry.primitives.point_cloud import PointCloud


def load_point_cloud(point_cloud_path, max_nr_points=None, verbose=False):
    """Loads point cloud from file.
    If the file is a mesh, points are its vertices.

    Args:
        point_cloud_path: point cloud file path
        max_nr_points: maximum number of points to load

    Returns:
        points_3d (N, 3): numpy array
    """

    # if exists, load it
    if os.path.exists(point_cloud_path):
        # if format is .ply or .obj
        if point_cloud_path.endswith(".ply") or point_cloud_path.endswith(".obj"):
            if verbose:
                print("loading point cloud from {}".format(point_cloud_path))
            point_cloud = o3d.io.read_point_cloud(point_cloud_path)
            points_3d = np.asarray(point_cloud.points)
            if max_nr_points is not None and points_3d.shape[0] > max_nr_points:
                # downsample
                random_idx = np.random.choice(
                    points_3d.shape[0], max_nr_points, replace=False
                )
                points_3d = points_3d[random_idx]
            if verbose:
                print("loaded {} points".format(points_3d.shape[0]))
            return points_3d
        else:
            print_error("unsupported point cloud format")
    else:
        print_error("point cloud path {} does not exist".format(point_cloud_path))


def load_point_clouds(point_clouds_paths, max_nr_points=10000, verbose=False):
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
        points_3d = load_point_cloud(pc_path, max_nr_points, verbose=verbose)
        point_cloud = PointCloud(points_3d)
        point_clouds.append(point_cloud)
    return point_clouds
