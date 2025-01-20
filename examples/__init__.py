import torch
import sys
import numpy as np
from pathlib import Path
import random
from typing import List, Union, Tuple, Optional, Type
from mvdatasets.utils.printing import print_error
from dataclasses import dataclass, field
import sys
import traceback
from rich import print


def custom_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Custom exception handler to print detailed information for uncaught exceptions.
    """
    # if issubclass(exc_type, KeyboardInterrupt):
    #     # Allow program to exit quietly on Ctrl+C
    #     sys.__excepthook__(exc_type, exc_value, exc_traceback)
    #     return

    # destroy_context()  # Clean up Dear PyGui

    # Format the exception message
    # message = f"{exc_type.__name__}: {exc_value}"
    # Pass detailed exception info to the print_error function
    # print_error(message, exc_type, exc_value, exc_traceback)

    # print(f"[bold red]ERROR:[/bold red] {message}")
    if exc_type and exc_traceback:
        print("\n[bold blue]Stack Trace:[/bold blue]")
        # Format the traceback into a readable format
        detailed_traceback = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        print(f"[dim]{detailed_traceback}[/dim]")


def get_dataset_test_preset(dataset_name: str = "dtu") -> Tuple[str, List[str], dict]:

    # test dtu
    if dataset_name == "dtu":
        scene_name = "dtu_scan83"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]

    # test blended-mvs
    elif dataset_name == "blended-mvs":
        scene_name = "bmvs_bear"
        splits = ["train", "test"]
        pc_paths = []

    # test nerf_synthetic
    elif dataset_name == "nerf_synthetic":
        scene_name = "lego"
        splits = ["train", "test"]
        pc_paths = ["tests/assets/point_clouds/nerf_synthetic/lego.ply"]

    # test shelly
    elif dataset_name == "shelly":
        scene_name = "khady"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]

    # test nerf_furry
    elif dataset_name == "nerf_furry":
        scene_name = "plushy"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]

    # test dmsr
    elif dataset_name == "dmsr":
        scene_name = "dinning"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]

    # test refnerf
    elif dataset_name == "refnerf":
        scene_name = "car"
        splits = ["train", "test"]
        pc_paths = []

    # test ingp
    elif dataset_name == "ingp":
        scene_name = "fox"
        splits = ["train", "test"]
        pc_paths = []

    # test llff
    elif dataset_name == "llff":
        scene_name = "fern"
        splits = ["train", "test"]
        pc_paths = ["tests/assets/point_clouds/llff/fern.ply"]

    # test mipnerf360
    elif dataset_name == "mipnerf360":
        scene_name = "garden"
        splits = ["train", "test"]
        pc_paths = []

    # test d-nerf
    elif dataset_name == "d-nerf":
        scene_name = "bouncingballs"
        splits = ["train", "test"]
        pc_paths = []

    # test visor
    elif dataset_name == "visor":
        scene_name = "P01_01"
        splits = ["train", "val"]
        pc_paths = []

    # test neu3d
    elif dataset_name == "neu3d":
        scene_name = "coffee_martini"
        splits = ["train", "val"]
        pc_paths = []

    # test panoptic-sports
    elif dataset_name == "panoptic-sports":
        scene_name = "basketball"
        splits = ["train", "test"]
        pc_paths = []

    # test iphone
    elif dataset_name == "iphone":
        scene_name = "paper-windmill"
        splits = ["train", "val"]
        pc_paths = []

    # test monst3r
    elif dataset_name == "monst3r":
        scene_name = "car-turn"
        splits = ["train"]
        pc_paths = []

    else:
        print_error(f"Dataset {dataset_name} does not have a test preset.")

    return {"scene_name": scene_name, "splits": splits, "pc_paths": pc_paths}
