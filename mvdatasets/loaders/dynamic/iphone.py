from rich import print
from pathlib import Path


def load(
    dataset_path: Path,
    scene_name: str,
    splits: list[str] = ["train", "test"],
    config: dict = {},
    verbose: bool = False,
):
    scene_path = dataset_path / scene_name
    pass
