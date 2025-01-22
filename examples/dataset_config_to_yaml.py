import tyro
import sys
from typing import List
from pathlib import Path
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.printing import print_error, print_warning, print_success
from mvdatasets.configs.example_config import ExampleConfig
from mvdatasets.io import load_yaml, save_yaml
from examples import get_dataset_test_preset, custom_exception_handler


def main(cfg: ExampleConfig, pc_paths: List[Path]):

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
        config=cfg.data.asdict(),
        point_clouds_paths=pc_paths,
        verbose=True,
    )

    # Save to a YAML file
    save_yaml(cfg.data.asdict(), output_path / "data_config.yaml")
    print_success("Config saved to file")

    # Load from the YAML file
    cfg_dict = load_yaml(output_path / "data_config.yaml")
    print_success("Config loaded from file")
    print(cfg_dict)

    print("done")


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
