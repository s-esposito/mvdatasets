import open3d as o3d
import numpy as np
from typing import Optional, List
from mvdatasets.geometry.projections import local_inv_perspective_projection


class Frustum:
    def __init__(self, line_set):
        self.line_set = line_set


def create_frustum(
    width: int,
    height: int,
    intrinsics: np.ndarray,
    frustum_color: Optional[List[float]] = None,
    size: float = 1.0,
):
    if frustum_color is None:
        frustum_color = [0, 1, 0]  # Default color

    # Validate intrinsics shape
    assert intrinsics.shape == (3, 3), "Intrinsics matrix must be 3x3."

    # Extract camera intrinsic parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    # cx = intrinsics[0, 2]
    # cy = intrinsics[1, 2]

    # Define 2D points in pixel coordinates
    points_2d_s = np.array(
        [
            [width, 0.0],
            [0.0, 0.0],
            [width, height],
            [0.0, height],
        ]
    )

    # Unproject 2D screen points to camera space
    points_3d_c = local_inv_perspective_projection(
        np.linalg.inv(intrinsics),  # Inverse of camera intrinsic matrix
        points_2d_s,
    )

    # multiply by depth
    depth = max(fx, fy) * 1e-3 * size  # Scale by `size`
    points_3d_c *= depth[..., None]

    # scale the frustum
    points_3d_c = points_3d_c * size
    points = points_3d_c

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frustum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set)

    return frustum
