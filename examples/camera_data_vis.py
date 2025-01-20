import tyro
import sys
import numpy as np
from examples import get_dataset_test_preset
from mvdatasets.mvdataset import MVDataset
from mvdatasets.visualization.matplotlib import plot_cameras_2d
from mvdatasets.configs.example_config import ExampleConfig
from examples import get_dataset_test_preset, custom_exception_handler
from mvdatasets.configs.example_config import ExampleConfig
from examples import get_dataset_test_preset, custom_exception_handler


def main(cfg: ExampleConfig):

    device = cfg.machine.device
    datasets_path = cfg.datasets_path
    output_path = cfg.output_path
    dataset_name = cfg.data.dataset_name
    scene_name = cfg.scene_name
    test_preset = get_dataset_test_preset(dataset_name)
    if scene_name is None:
        scene_name = test_preset["scene_name"]
    print("scene_name: ", scene_name)

    pc_paths = test_preset["pc_paths"]
    splits = test_preset["splits"]

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        splits=splits,
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
    sys.excepthook = custom_exception_handler
    args = tyro.cli(ExampleConfig)
    print(args)
    main(args)
