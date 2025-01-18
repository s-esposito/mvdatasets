# MVDatasets


<img align="right" width="130" height="130" src="imgs/art.webp">

### Standardized DataLoaders for 3D Computer Vision

[Stefano Esposito](https://s-esposito.github.io/), [Andreas Geiger](https://www.cvlibs.net/)
<br>
University of Tübingen, [Autonomous Vision Group](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/home/)

[![Unit tests](https://github.com/s-esposito/mvdatasets/actions/workflows/run-unit-tests.yml/badge.svg?branch=dev)](https://github.com/s-esposito/mvdatasets/actions/workflows/run-unit-tests.yml) [![deploy-documentation](https://github.com/s-esposito/mvdatasets/actions/workflows/deploy-docs.yml/badge.svg?branch=dev)](https://github.com/s-esposito/mvdatasets/actions/workflows/deploy-docs.yml)

```
⚠️ This is a work in progress research codebase designed with a focus on modularity; future updates *will try* not to disrupt existing functionalities.
```

---

Our goal is to provide a plug and play library to quickly develop and test new research ideas. We offer various data loaders for commonly used multi-view datasets in 3D reconstruction and view-synthesis, that work out of the box without further data processing.


<div style="display: flex; justify-content: center;">
<table style="text-align: left; border-collapse: collapse; width: 80%; margin: auto;">
<tr>
<td style="vertical-align: top; padding: 10px; border: 1px solid #ddd;">

**Static:**
- [x] [NeRF-Synthetic](https://www.matthewtancik.com/nerf): bounded
- [x] [Ref-NeRF](https://dorverbin.github.io/refnerf/): bounded
- [x] [Shelly](https://research.nvidia.com/labs/toronto-ai/adaptive-shells/): bounded
- [x] [DTU](https://github.com/lioryariv/idr?tab=readme-ov-file): unbounded
- [x] [BlendedMVS](https://github.com/Totoro97/NeuS?tab=readme-ov-file): unbounded
- [x] [Mip-NeRF360](https://jonbarron.info/mipnerf360/): unbounded
- [ ] [NeRF-LLFF](https://www.matthewtancik.com/nerf): unbounded
- [ ] ...

</td>
<td style="vertical-align: top; padding: 10px; border: 1px solid #ddd;">


**Dynamic:**
- [x] [D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html): semi-monocular, bounded
- [x] [PanopticSports](https://dynamic3dgaussians.github.io/): multi-view, bounded
- [ ] [Neu3D](https://github.com/facebookresearch/Neural_3D_Video): multi-view, unbounded
- [x] [VISOR](https://epic-kitchens.github.io/VISOR/): monocular, unbounded
- [ ] [Nerfies](https://github.com/google/nerfies/releases/tag/0.1): monocular, unbounded
- [ ] [Hypernerf](https://github.com/google/hypernerf/releases/tag/v0.1): monocular, unbounded
- [x] [iPhone](https://kair-bair.github.io/dycheck/): monocular, unbounded
- [x] [MonST3R](https://github.com/Junyi42/monst3r): monocular, unbounded
- [ ] [DynamicScenes aka. NVIDIA]([#](https://gorokee.github.io/jsyoon/dynamic_synth/)): monocular, unbounded
- [ ] [AMA](https://people.csail.mit.edu/drdaniel/mesh_animation/#data): multi-view, bounded
- [ ] [watch-it-move](https://github.com/NVlabs/watch-it-move): multi-view, bounded
- [ ] ...

</td>
</tr>
</table>
</div>

## Cameras

We use the OpenCV camera coordinate system:
- X axis: Points to the right of the camera's sensor. It extends horizontally from the left side to the right side of the image. Increasing values move towards the right side of the image.
- Y axis: Points downward from the camera's sensor. It extends vertically from the top to the bottom of the image. Increasing values move towards the bottom of the image.
- Z axis: Represents depth and points away from the camera lens. It extends from the camera's lens outward into the scene. Increasing values move away from the camera.

<p float="left">
  <img src="imgs/pose_and_intrinsics.png" width="500"/>
  <img src="imgs/projection_with_principal_point_offset.png" width="320"/>
</p>

Images taken from Andreas Geiger's Computer Vision [lectures](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/) at the University of Tübingen.

## Installation

Install the module with `pip install mvdatasets`, or from source using `pip install -e`.

## Run the examples

```bash
# download data in ./data
bash scripts/download/nerf_synthetic.sh
# visualize dataset splits
python examples/dataset_splits_vis.py.py --dataset-name nerf_synthetic --datasets-path ./data
```

<p float="left">
  <img src="imgs/blender_train_cameras.png" width="400"/>
  <img src="imgs/blender_test_cameras.png" width="400"/>
</p>

## Disclaimer

Functions located in any `.deprecated` folder may no longer work as expected. While they might be supported again in the future, this is not guaranteed.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code as long as you provide proper attribution to the original author(s).

## Citation

If you use this library for your research, please consider citing:

```bibtex
@misc{Esposito2024MVDatasets,
  author       = {Stefano Esposito and Andreas Geiger},
  title        = {MVDatasets: Standardized DataLoaders for 3D Computer Vision},
  year         = {2025},
  url          = {https://github.com/s-esposito/mvdatasets},
  note         = {GitHub repository}
}
```