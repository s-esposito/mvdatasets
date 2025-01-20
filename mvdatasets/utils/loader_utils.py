from typing import List, Tuple
import numpy as np


def get_min_max_cameras_distances(poses: list) -> tuple:
    """
    return maximum pose distance from origin

    Args:
        poses (list): list of numpy (4, 4) poses

    Returns:
        min_dist (float): minumum camera distance from origin
        max_dist (float): maximum camera distance from origin
    """
    if len(poses) == 0:
        raise ValueError("poses list empty")

    # get all camera centers
    camera_centers = np.stack(poses, 0)[:, :3, 3]
    camera_distances_from_origin = np.linalg.norm(camera_centers, axis=1)

    min_dist = np.min(camera_distances_from_origin)
    max_dist = np.max(camera_distances_from_origin)

    return min_dist, max_dist


def rescale(
    all_poses: List[np.ndarray], to_distance: float = None
) -> Tuple[float, float, float]:
    """returns a scaling factor for the scene such that the furthest camera is at target distance

    Args:
        all_poses (List[np.ndarray]): list of camera poses
        to_distance (float, optional): maximum distance of the furthest camera from the origin. Defaults to None.

    Returns:
        float: scaling factor
        float: distance of rescaled closest camera from the origin
        float: distance of rescaled furthest camera from the origin
    """
    # init multiplier
    scene_radius_mult = 1.0

    # find scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(all_poses)

    if to_distance is None:
        return scene_radius_mult, min_camera_distance, max_camera_distance

    # scene scale such that furthest away camera is at target distance
    scene_radius_mult = to_distance / max_camera_distance

    # new scene scale
    min_camera_distance = min_camera_distance * scene_radius_mult
    max_camera_distance = max_camera_distance * scene_radius_mult

    return scene_radius_mult, min_camera_distance, max_camera_distance
