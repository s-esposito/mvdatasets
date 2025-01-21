Camera
======

Coordinate system
-----------------

Central class of the library; it provides a standardized format for camera parameters, its data and methods to manipulate them.
All data in stored in CPU.

We use the OpenCV camera coordinate system:

- X axis: Points to the right of the camera's sensor. It extends horizontally from the left side to the right side of the image. Increasing values move towards the right side of the image.
- Y axis: Points downward from the camera's sensor. It extends vertically from the top to the bottom of the image. Increasing values move towards the bottom of the image.
- Z axis: Represents depth and points away from the camera lens. It extends from the camera's lens outward into the scene. Increasing values move away from the camera.

Camera
------

.. automodule:: mvdatasets.camera
   :members:
   :undoc-members: