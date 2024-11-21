#! /bin/bash

DATASETS=(
    "ingp"
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
    python tests/vis_camera_rays.py --dataset-name $dataset
    python tests/train_test_splits.py --dataset-name $dataset
    python tests/pixels_sampling.py --dataset-name $dataset
    python tests/points_projections.py --dataset-name $dataset
    python tests/tensor_reel.py --dataset-name $dataset
    python tests/bounding_volumes.py --dataset-name $dataset
done