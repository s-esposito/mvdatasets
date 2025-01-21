import tyro
import sys
import numpy as np
from pathlib import Path
from examples import get_dataset_test_preset
from mvdatasets.mvdataset import MVDataset
from mvdatasets.visualization.matplotlib import plot_cameras_2d
from mvdatasets.configs.example_config import ExampleConfig
from examples import get_dataset_test_preset, custom_exception_handler
from mvdatasets.utils.printing import print_warning


def main(cfg: ExampleConfig, pc_paths: list[Path]):

    device = cfg.machine.device
    datasets_path = cfg.datasets_path
    output_path = cfg.output_path
    scene_name = cfg.scene_name
    dataset_name = cfg.data.dataset_name

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        config=cfg.data,
        point_clouds_paths=pc_paths,
        verbose=True,
    )

    # # random camera index
    # rand_idx = np.random.randint(0, len(mv_data.get_split("test")))
    # camera = deepcopy(mv_data.get_split("test")[rand_idx])

    plot_cameras_2d(
        cameras=mv_data.get_split("train"),
    )


if __name__ == "__main__":

    # custom exception handler
    sys.excepthook = custom_exception_handler

    # parse arguments
    args = tyro.cli(ExampleConfig)

    # get test preset
    test_preset = get_dataset_test_preset(args.data.dataset_name)
    # scene name
    if args.scene_name is None:
        args.scene_name = test_preset["scene_name"]
        print_warning(
            f"scene_name is None, using preset test scene {args.scene_name} for dataset"
        )
    # additional point clouds paths (if any)
    pc_paths = test_preset["pc_paths"]

    # start the example program
    main(args, pc_paths)
