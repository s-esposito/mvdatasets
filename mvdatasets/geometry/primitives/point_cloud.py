import numpy as np
from typing import Optional
from mvdatasets.geometry.rigid import apply_transformation_3d


class PointCloud:

    def __init__(
        self,
        points_3d: np.ndarray,
        points_rgb: Optional[np.ndarray] = None,  # (N, 3) or (3,)
        color: Optional[str] = None,
        label: Optional[str] = None,
        size: Optional[float] = None,
        marker: Optional[str] = None,
    ):
        self.points_3d = points_3d
        self.points_rgb = points_rgb

        if self.points_rgb is not None:
            # check if dimensions are correct
            if self.points_rgb.ndim == 2:
                # first dimension must be the same as points_3d
                if self.points_rgb.shape[0] != self.points_3d.shape[0]:
                    raise ValueError(
                        f"Points RGB must have the same number of points as points 3D, got {self.points_rgb.shape[0]} and {self.points_3d.shape[0]}"
                    )
                # second dimension must be 3
                if self.points_rgb.shape[1] != 3:
                    raise ValueError(
                        f"Points RGB must have shape (N, 3), got {self.points_rgb.shape}"
                    )
            elif self.points_rgb.ndim == 1:
                # first dimension must be 3
                if self.points_rgb.shape[0] != 3:
                    raise ValueError(
                        f"Points RGB must have shape (3,), got {self.points_rgb.shape}"
                    )
            else:
                raise ValueError(
                    f"Points RGB must have shape (N, 3) or (3,), got {self.points_rgb.shape}"
                )

        # plotting attributes
        self.color = color
        self.label = label
        self.size = size
        self.marker = marker

    def downsample(self, nr_points: int):
        if nr_points >= self.points_3d.shape[0]:
            # do nothing
            return

        idxs = np.random.choice(self.points_3d.shape[0], nr_points, replace=False)
        self.points_3d = self.points_3d[idxs]

        if self.points_rgb is not None:
            self.points_rgb = self.points_rgb[idxs]

    def mask(self, mask: np.ndarray):
        self.points_3d = self.points_3d[mask]

        if self.points_rgb is not None:
            self.points_rgb = self.points_rgb[mask]

    def shape(self):
        return self.points_3d.shape

    def __str__(self) -> str:
        return f"PointCloud with {self.points_3d.shape[0]} points"

    def transform(self, transformation: np.ndarray):
        self.points_3d = apply_transformation_3d(self.points_3d, transformation)
