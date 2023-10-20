# DataSets: standardized loaders for multi-view datasets

[Stefano Esposito](https://s-esposito.github.io/),
<br>
University of Tübingen, Autonomous Vision Group

Our goal is to provide a plug and play library to quickly develop and test new research ideas. We offer various data loaders for commonly used multi-view datasets in 3D reconstruction and view-synthesis, that work out of the box without further data processing.

Static (3D):
- [DTU](#)
- [NeRF-Synthetic](#)

Dynamic (4D):
- [PAC-NERF](#)

Soon to be supported:
- [NeRF-LLFF](#)
- [NeRF-360](#)

## Cameras
This code uses an "OpenGL" style camera coordinate system:
- *Right Vector* ($`x`$): Represents rightward direction.
- *Up Vector* ($`y`$): Represents upward direction.
- *Forward Vector* ($`z`$): Represents backward direction (along negative Z-axis).
    
<p align="middle">
  <img src="imgs/datasets_frame.png" width="400"/>
</p>

<p align="middle">
  <img src="imgs/dtu_poses.png" width="600"/>
</p>

<p align="middle">
  <img src="imgs/data_loader.gif" width="600"/>
</p>

<!---

# Citation

If you use this library for your research, please consider citing:

```
@inproceedings{datasets,
	title        = {DataSets: Standardized Loaders for Multi-View Datasets},
	author       = {
		Stefano Esposito
	},
	year         = 2023
}
```

# Contributors

<a href="https://github.com/s-esposito/datasets/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=s-esposito/datasets" />
</a>

-->