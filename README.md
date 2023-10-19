# DataSets: standardized loaders for multi-view datasets

[Stefano Esposito](https://s-esposito.github.io/),
<br>
University of Tübingen, Autonomous Vision Group

Various data loaders for multi-view datasets commonly used in 3D reconstruction and view-synthesis.

Static:
- [DTU](#)
- [NeRF-Synthetic](#)

Dynamica:
- [PAC-NERF](#)

Soon to be supported:
- [NeRF-LLFF](#)
- [NeRF-360](#)

## Misc
This code uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), the X-axis points right, and the Z-axis points into the image plane.
    
<p align="middle">
  <img src="imgs/datasets_frame.png" width="400"/>
</p>