import numpy as np
import math
from mvdatasets import Camera
from mvdatasets.geometry.common import look_at

EPS = 1e-3


def _get_intrinsics(height, width, fovy):
    # intrinsics
    focal = height / (2 * np.tan(np.radians(fovy) / 2))
    intrinsics = np.eye(3)
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = width / 2
    intrinsics[1, 2] = height / 2
    return intrinsics


def _get_pose(azimuth_deg, elevation_deg, radius, center, up):
    # extrinsics
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    # convert spherical coordinates (radius, azimuth, elevation) to position
    position = np.array(
        [
            radius * math.cos(elevation_rad) * math.sin(azimuth_rad),  # x
            radius * math.cos(elevation_rad) * math.cos(azimuth_rad),  # y
            radius * math.sin(elevation_rad),  # z
        ],
        dtype=np.float32,
    )

    return look_at(position, center, up)


class OrbitCamera(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        radius: float = 1.0,
        fovy: float = 45.0,
        near: float = 0.1,
        far: float = 100,
        center: np.ndarray = np.array(
            [0, 0, 0], dtype=np.float32
        ),  # look at this point
        up: str = "z",
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
    ):
        if up == "z":
            up = np.array([0, 0, 1], dtype=np.float32)
        elif up == "y":
            up = np.array([0, 1, 0], dtype=np.float32)
        else:
            raise ValueError(f"Invalid `up` value: {up}, must be 'z' or 'y'.")

        # init base class
        # compute pose and intrinsics
        intrinsics = _get_intrinsics(height, width, fovy)
        pose = _get_pose(azimuth_deg, elevation_deg, radius, center, up)
        super().__init__(
            intrinsics=intrinsics,
            pose=pose,
            width=width,
            height=height,
            near=near,
            far=far,
        )
        # orbit camera parameters
        self.radius = radius  # camera distance from center
        self.fovy = fovy  # in degree
        if not isinstance(center, np.ndarray):
            center = np.array(center, dtype=np.float32)
        self.center = center  # look at this point
        self.up = np.array(up, dtype=np.float32)
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = elevation_deg

    def _update(self):
        intrinsics = _get_intrinsics(
            self.height, self.width, self.fovy
        )
        self.set_intrinsics(intrinsics)
        self.pose = _get_pose(
            self.azimuth_deg, self.elevation_deg, self.radius, self.center, self.up
        )

    # setters

    def set_center(self, center):
        self.center = center
        self._update()

    def set_fov(self, fovy):
        self.fovy = fovy
        self._update()

    def set_elevation_deg(self, elevation_deg):
        self.elevation_deg = np.clip(elevation_deg, -80 + EPS, 80 - EPS)
        self._update()

    def set_azimuth_deg(self, azimuth_deg):
        self.azimuth_deg = azimuth_deg
        self._update()

    def set_radius(self, radius):
        self.radius = np.clip(radius, EPS, np.inf)
        self._update()

    def orbit(self, dx, dy):
        self.azimuth_deg += 0.005 * dx
        elevation_deg = self.elevation_deg + 0.005 * dy
        # clipping
        self.elevation_deg = np.clip(elevation_deg, -80 + EPS, 80 - EPS)
        # update
        self._update()

    def move_up(self):
        # elevation angle between 0 and 180 degrees
        elevation_deg = self.elevation_deg + 1
        self.elevation_deg = np.clip(elevation_deg, -80 + EPS, 80 - EPS)
        # update
        self._update()

    def move_down(self):
        # elevation angle between 0 and 180 degrees
        elevation_deg = self.elevation_deg - 1
        self.elevation_deg = np.clip(elevation_deg, -80 + EPS, 80 - EPS)
        # update
        self._update()

    def move_left(self):
        self.azimuth_deg += 1
        # update
        self._update()

    def move_right(self):
        self.azimuth_deg -= 1
        # update
        self._update()

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)
        # update
        self._update()

    # def pan(self, dx, dy, dz=0):
    #     # pan in camera coordinate system (careful on the sensitivity!)
    #     self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])
