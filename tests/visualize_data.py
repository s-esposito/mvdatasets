import tyro
import numpy as np
import os
from copy import deepcopy
from config import get_dataset_test_preset
from config import Args
from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.utils.printing import print_error, print_warning
from mvdatasets.visualization.matplotlib import plot_cameras_2d


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    test_preset = get_dataset_test_preset(dataset_name)
    scene_name = test_preset["scene_name"]
    pc_paths = test_preset["pc_paths"]
    config = test_preset["config"]
    splits = test_preset["splits"]

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=splits,
        config=config,
        verbose=True,
    )

    # # random camera index
    # rand_idx = np.random.randint(0, len(mv_data.get_split("test")))
    # camera = deepcopy(mv_data.get_split("test")[rand_idx])

    plot_cameras_2d(
        cameras=mv_data.get_split("test"),
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
