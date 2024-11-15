# TODO: add tests for this functions

import numpy as np

# horizontal coordinate system to cartesian coordinate system
# azimuth: 0 to 360
# elevation: -90 to 90


class SphereSpiral:

    def __init__(self, turns=5, points_per_turn=20):
        self.turns = turns
        self.points_per_turn = points_per_turn
        self.points = self.create_points_trajectory()

    def _spiral(self, azimuth_deg, elevation_deg):

        # azimuth_min = 0, azimuth_max = 360
        # elevation_min = -90, elevation_max = 90

        def deg2rad(deg):
            return deg * (np.pi / 180)

        azimuth_rad = deg2rad(azimuth_deg)
        elevation_rad = deg2rad(elevation_deg)

        x = np.cos(azimuth_rad) * np.cos(elevation_rad)
        y = np.sin(azimuth_rad) * np.cos(elevation_rad)
        z = np.sin(elevation_rad)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        return np.column_stack((x, y, z))

    def create_points_trajectory(self):

        # points can be seen as unit vectors pointing from the origin to a point on the surface of the sphere

        p = self._spiral(
            azimuth_deg=np.tile(
                np.linspace(1, 360, self.points_per_turn), self.turns
            ),  # angle around the z-axis
            elevation_deg=np.linspace(
                -90, 90, self.turns * self.points_per_turn
            ),  # angle from the xy-plane
        )

        return p  # (turns * points_per_turn, 3)


# spiral_points = SphereSpiral(turns=4, points_per_turn=10).points
# plot_points_trajectory(spiral_points, show_plot=False, save_fig=True, save_dir="plots")
# view_dirs = -1 * spiral_points
