#! /bin/bash

DATASETS=(
    "dtu"
    "blender"
    "blendernerf"
    "dmsr"
    "refnerf"
    "llff"
    "mipnerf360"
    "shelly"
)

for dataset in "${DATASETS[@]}"; do
    echo "#### testing $dataset ####"
    python tests/train_test_splits.py $dataset
    python tests/pixels_sampling.py $dataset
    python tests/camera_rays.py $dataset
    python tests/reproject_points.py $dataset
    python tests/tensor_reel.py $dataset
done