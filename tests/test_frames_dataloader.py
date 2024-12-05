import tyro
import torch
import numpy as np
import os
from tqdm import tqdm
from typing import List
from config import get_dataset_test_preset
from config import Args
from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.utils.printing import print_error, print_warning
from mvdatasets import Camera


class DataSplit:

    def __init__(
        self,
        cameras: List[Camera],
        modalities: list[str] = ["rgbs", "masks"],
    ):
        self.nr_cameras = len(cameras)
        # assumption: all cameras have the same dimensions
        self.temporal_dim = cameras[0].get_temporal_dim()
        self.width = cameras[0].get_width()
        self.height = cameras[0].get_height()

        data = {}
        intrinsics_all = []
        intrinsics_inv_all = []
        c2w_all = []
        w2c_all = []
        timestamps_all = []

        for camera in cameras:
            
            # get camera data
            for key, val in camera.data.items():
                # populate data dict
                if key in modalities:
                    if key not in data:
                        data[key] = []
                    if val is not None:
                        data[key].append(val)
                    else:
                        print_error(f"camera {camera.camera_idx} has no {key} data")
            
            intrinsics_all.append(camera.get_intrinsics())
            intrinsics_inv_all.append(camera.get_intrinsics_inv())
            c2w_all.append(camera.get_pose())
            w2c_all.append(camera.get_pose_inv())
            timestamps_all.append(camera.get_timestamps())

        self.intrinsics_all = np.stack(intrinsics_all)  # (N, 3, 3)
        self.intrinsics_inv_all = np.stack(intrinsics_inv_all)  # (N, 3, 3)
        self.c2w_all = np.stack(c2w_all)  # (N, 4, 4)
        self.w2c_all = np.stack(w2c_all)  # (N, 4, 4)
        self.timestamps_all = np.stack(timestamps_all)  # (N, T)

        # concat data and move to device
        for key, val in data.items():
            data[key] = np.stack(val)  # (N, T, H, W, C)
        self.data = data

        print(f"DataSplit initialized with {len(cameras)} cameras.")

    def __len__(self):
        # returns the number of cameras frames in the split
        return self.nr_cameras * self.temporal_dim

    def __getitem__(self, idx) -> Camera:

        # index indexes all cameras frames in the split [0, N_cam * T]
        # get camera id by dividing by the temporal dimension
        cam_idx = idx // self.temporal_dim
        frame_idx = idx % self.temporal_dim
        
        data = {
            "intrinsics": torch.from_numpy(self.intrinsics_all[cam_idx]).float(),  # (3, 3)
            "intrinsics_inv": torch.from_numpy(self.intrinsics_inv_all[cam_idx]).float(),  # (3, 3)
            "c2w": torch.from_numpy(self.c2w_all[cam_idx]).float(),  # (4, 4)
            "w2c": torch.from_numpy(self.w2c_all[cam_idx]).float(),  # (4, 4)
            "timestamps": torch.tensor(self.timestamps_all[cam_idx, frame_idx]).float()  # (1,)
        }
        
        for k, v in self.data.items():
            # keep dtype consistent with the one in Camera.data
            data[k] = torch.from_numpy(v[cam_idx, frame_idx])  # (H, W, C)
        
        return data

    def __str__(self) -> str:
        return f"DataSplit with {len(self)} cameras, totalling {self.get_memory_footprint()} bytes."
    
    def get_memory_footprint(self) -> int:
        # returns the memory footprint of the split in bytes
        memory_footprint = 0
        for key, val in self.data.items():
            memory_footprint += val.nbytes
        memory_footprint += self.intrinsics_all.nbytes
        memory_footprint += self.intrinsics_inv_all.nbytes
        memory_footprint += self.c2w_all.nbytes
        memory_footprint += self.w2c_all.nbytes
        memory_footprint += self.timestamps_all.nbytes
        return memory_footprint


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
        pose_only=False,
        verbose=True,
    )

    batch_size = 32
    data_loaders = {}
    # for split_name, cameras_list in mv_data.data.items():
    split_name = "train"
    cameras_list = mv_data.get_split(split_name)
    # initialize data loader
    data_split = DataSplit(
        cameras=cameras_list,
        modalities=["rgbs", "masks"],
        )
    # 
    data_loader = torch.utils.data.DataLoader(
        data_split,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        persistent_workers=True,  # Reduce worker initialization overhead
        pin_memory=True,  # For faster transfer to GPU
    )
    data_loaders[split_name] = data_loader
    #
    print(f"Data loader for {split_name} initialized.")
    
    # test loop
    
    nr_epochs = 2
    data_loader = data_loaders["train"]
    
    # Loop over epochs
    epochs_pbar = tqdm(range(nr_epochs), desc="epochs")
    for epoch_nr in epochs_pbar:
        # Get iterator over data
        dataiter = iter(data_loader)
        # Loop over batches
        iter_pbar = tqdm(range(len(dataiter)), desc="iter")
        for iter_nr in iter_pbar:
            
            try:
                # Fetch the next batch
                batch = next(dataiter)
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                # Print batch shape
                for k, v in batch.items():
                    print(f"{epoch_nr}, {iter_nr}, {k}: {v.shape}, {v.dtype}, {v.device}")
                    
            except StopIteration:
                # Handle iterator exhaustion (shouldn't occur unless manually breaking loop)
                break


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
