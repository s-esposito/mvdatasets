# MV Datasets: standardized loaders for multi-view datasets

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
The camera coordinate system is the OpenCV one (right-handed):
- X-Axis: Points to the right of the camera's sensor. It extends horizontally from the left side to the right side of the image. Increasing values move towards the right side of the image.
- Y-Axis: Points downward from the camera's sensor. It extends vertically from the top to the bottom of the image. Increasing values move towards the bottom of the image.
- Z-Axis: Represents depth and points away from the camera lens. It extends from the camera's lens outward into the scene. Increasing values move away from the camera.

<p float="left">
  <img src="imgs/pose_and_intrinsics.png" width="500"/>
  <img src="imgs/projection_with_principal_point_offset.png" width="320"/>
</p>   

Images taken from Andreas Geiger's Computer Vision [lectures](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/) at the University of Tübingen.

<!-- 
<p align="middle">
  <img src="imgs/datasets_frame.png" width="400"/>
</p>
-->

### Train and test splits

<p float="left">
  <img src="imgs/blender_training_cameras.png" width="400" />
  <img src="imgs/blender_test_cameras.png" width="400" />
</p>


<!--
TODO: update 
<p align="middle">
  <img src="imgs/dtu_poses.png" width="600"/>
</p>

<p align="middle">
  <img src="imgs/data_loader.gif" width="600"/>
</p>
-->

## Installation

```bash
# 1) install requirements
todo

# 2) install library
python setup.py develop
pip install dist/mvdatasets-0.2.tar.gz 
# or
python setup.py develop
```

<!-- 
TODO: (optional: install PyTorch from source with CXX11_ABI=1) 


```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

Get the PyTorch Source

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

Install PyTorch

On Linux

Install [ccache](https://github.com/ccache/ccache/blob/master/doc/INSTALL.md) (optional)
```bash
# download bin and add it to path or compile source and install
```

Prepare conda env
```bash
conda install -c conda-forge cusparselt
conda install -c anaconda cudnn
```

If you would like to compile PyTorch with new C++ ABI enabled, then first run this [commands](https://discuss.pytorch.org/t/how-to-specify-glibcxx-use-cxx11-abi-1-when-building-pytorch-from-source/102899):

```bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
python setup.py develop
```

Invoke CMake to Build C++ Code:

First, make sure you invoke CMake to build your C++ code and create the shared library (e.g., .so file on Unix systems). From your project directory, run:


```bash
mkdir build
cd build
cmake ..
make
```
This step will generate the shared library containing your C++ functions.

Run install:

After you have successfully built the C++ code using CMake, you can install the Python package using python setup.py install. This step will package the compiled C++ library along with the Python interface code and install it as a Python package.

```bash
python setup.py develop 
```
-->

## Testing

```bash
# reproduce images
python tests/train_test_splits.py
python tests/pixels_sampling.py
python tests/camera_rays.py
python tests/reproject_points.py
python tests/tensor_reel.py
```

## Todo

- [ ] sample probability based on training error

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