import tyro
import torch
import numpy as np
import os
from tqdm import tqdm
from typing import List
from examples import get_dataset_test_preset
from examples import Args
from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.utils.printing import print_error, print_warning, print_success
from mvdatasets.utils.memory import bytes_to_gb
from mvdatasets import Camera
from mvdatasets.utils.raycasting import get_pixels
from mvdatasets import Profiler
from mvdatasets import DataSplit


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    scene_name = args.scene_name
    test_preset = get_dataset_test_preset(dataset_name)
    if scene_name is None:
        scene_name = test_preset["scene_name"]
    pc_paths = test_preset["pc_paths"]
    splits = test_preset["splits"]

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=splits,
        pose_only=False,
        verbose=True,
    )

    nr_epochs = 10
    num_workers = os.cpu_count() // 2
    print(f"num_workers: {num_workers}")
    shuffle = True
    persistent_workers = True  # Reduce worker initialization overhead
    pin_memory = True  # For faster transfer to GPU
    batch_size = 4  # nr full frames per batch
    print(f"batch_size: {batch_size}")
    
    benchmark = True

    # -------------------------------------------------------------------------

    nr_sequence_frames = 0
    cameras_temporal_dim = mv_data.get_split("train")[0].get_temporal_dim()
    if cameras_temporal_dim > 1:
        use_incremental_sequence_lenght = True
    else:
        use_incremental_sequence_lenght = False
    increase_nr_sequence_frames_each = 1

    # index frames -------------------------------------------------------------

    run_index_frames = True
    if run_index_frames:
        
        if benchmark:
            # Set profiler
            profiler = Profiler()  # nb: might slow down the code
        else:
            profiler = None

        # Loop over epochs
        step = 0
        epochs_pbar = tqdm(range(nr_epochs), desc="epochs", ncols=100)
        for epoch_nr in epochs_pbar:
            
            if profiler is not None:
                profiler.start("epoch")
            
            # if first iteration or need to update time dimension of data split
            if (
                epoch_nr == 0  # first iteration
                # need to update time dimension of data split
                or (
                    use_incremental_sequence_lenght
                    and nr_sequence_frames < cameras_temporal_dim
                    and epoch_nr % increase_nr_sequence_frames_each == 0
                )
            ):

                nr_sequence_frames += 1

                # initialize data loader
                data_split = DataSplit(
                    cameras=mv_data.get_split("train"),
                    nr_sequence_frames=nr_sequence_frames,
                    modalities=mv_data.get_split_modalities("train"),
                    index_pixels=False,  # index frames (!!!)
                )

                #
                data_loader = torch.utils.data.DataLoader(
                    data_split,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    persistent_workers=persistent_workers,
                    pin_memory=pin_memory,
                )

            # Iterate over batches
            iter_pbar = tqdm(data_loader, desc="iter", ncols=100, disable=False)
            for iter_nr, batch in enumerate(iter_pbar):
                
                if profiler is not None:
                    profiler.start("iter")
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                if not benchmark:
                    # Print batch shape
                    for k, v in batch.items():
                        print(
                            f"{epoch_nr}, {iter_nr}, {k}: {v.shape}, {v.dtype}, {v.device}"
                        )

                # Increment step
                step += 1
                
                if profiler is not None:
                    profiler.end("iter")
                
            if profiler is not None:
                profiler.end("epoch")
                
        print_success(f"Done after {step} steps")
        
        if profiler is not None:
            profiler.print_avg_times()

    # index pixels (too slow) -------------------------------------------------

    run_index_pixels = False
    if run_index_pixels:
        
        if benchmark:
            # Set profiler
            profiler = Profiler()  # nb: might slow down the code
        else:
            profiler = None

        # initialize data loader
        data_split = DataSplit(
            cameras=mv_data.get_split("train"),
            modalities=mv_data.get_split_modalities("train"),
            index_pixels=True,
        )

        batch_size = 512  # nr of rays sampled per batch
        print(f"batch_size: {batch_size}")

        # test loop

        # Loop over epochs
        step = 0
        epochs_pbar = tqdm(range(nr_epochs), desc="epochs", ncols=100)
        for epoch_nr in epochs_pbar:
            
            if profiler is not None:
                profiler.start("epoch")
            
            # if first iteration or need to update time dimension of data split
            if (
                epoch_nr == 0  # first iteration
                # need to update time dimension of data split
                or (
                    use_incremental_sequence_lenght
                    and nr_sequence_frames < cameras_temporal_dim
                    and epoch_nr % increase_nr_sequence_frames_each == 0
                )
            ):

                nr_sequence_frames += 1

                # initialize data loader
                data_split = DataSplit(
                    cameras=mv_data.get_split("train"),
                    nr_sequence_frames=nr_sequence_frames,
                    modalities=mv_data.get_split_modalities("train"),
                    index_pixels=True,  # index pixels (!!!)
                )

                #
                data_loader = torch.utils.data.DataLoader(
                    data_split,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    persistent_workers=persistent_workers,
                    pin_memory=pin_memory,
                )
            
            # Iterate over batches
            iter_pbar = tqdm(data_loader, desc="iter", ncols=100, disable=False)
            for iter_nr, batch in enumerate(iter_pbar):
                
                if profiler is not None:
                    profiler.start("iter")

                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                if not benchmark:
                    # Print batch shape
                    for k, v in batch.items():
                        print(
                            f"{epoch_nr}, {iter_nr}, {k}: {v.shape}, {v.dtype}, {v.device}"
                        )
                
                # Increment step
                step += 1
                
                if profiler is not None:
                    profiler.end("iter")
                
            if profiler is not None:
                profiler.end("epoch")
                
        print_success(f"Done after {step} steps")

        if profiler is not None:
            profiler.print_avg_times()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
