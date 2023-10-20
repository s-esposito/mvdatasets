#include "datasets/PyBridge.h"

#include "torch/torch.h"
#include "datasets/TensorReel.h"

namespace py = pybind11;

PYBIND11_MODULE(datasets, m)
{

    py::class_<TensorReel>(m, "TensorReel")
        .def(py::init())
        .def_readwrite("rgb_reel", &TensorReel::rgb_reel)
        .def_readwrite("mask_reel", &TensorReel::mask_reel)
        .def_readwrite("K_reel", &TensorReel::K_reel)
        .def_readwrite("tf_cam_world_reel", &TensorReel::tf_cam_world_reel);

    // py::class_<MiscDataFuncs>(m, "MiscDataFuncs")
    //     .def(py::init())
    //     .def_static("frames2tensors", &MiscDataFuncs::frames2tensors);
}