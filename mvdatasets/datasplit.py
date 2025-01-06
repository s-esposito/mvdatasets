import torch
import numpy as np
from typing import List
from mvdatasets.utils.printing import print_error, print_warning, print_success
from mvdatasets.utils.memory import bytes_to_gb
from mvdatasets import Camera
from mvdatasets.utils.raycasting import get_pixels


class DataSplit:

    def __init__(
        self,
        cameras: List[Camera],
        nr_sequence_frames: int = -1,
        modalities: list[str] = ["rgbs", "masks"],
        index_pixels: bool = False,
    ):
        """_summary_

        Args:
            cameras (List[Camera]): list of Camera objects.
            nr_sequence_frames (int, optional): Defaults to -1, means all temporal frames are used.
            modalities (list[str], optional): Defaults to ["rgbs", "masks"].
            index_pixels (bool, optional): If True, indexes images pixels directly. If False, indexes whole images. Defaults to False.
        """
        self.nr_cameras = len(cameras)
        # assumption: all cameras have the same dimensions (T)
        temporal_dim = cameras[0].get_temporal_dim()
        
        if nr_sequence_frames == -1:
            self.temporal_dim = temporal_dim
        else:
            # assert temporal_dim >= nr_sequence_frames
            if nr_sequence_frames > temporal_dim:
                print_warning(f"nr_sequence_frames: {nr_sequence_frames} > temporal_dim: {temporal_dim}, capping to {temporal_dim}")
                nr_sequence_frames = temporal_dim
            self.temporal_dim = nr_sequence_frames
        
        self.width = cameras[0].get_width()
        self.height = cameras[0].get_height()
        self.index_pixels = index_pixels

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
                        # val is (T, H, W, C)
                        # keep only the first temporal_dim frames
                        val_ = val[:self.temporal_dim]
                        # val_ is (T_s, H, W, C)
                        data[key].append(val_)
                    else:
                        print_error(f"camera {camera.camera_label} has no {key} data")

            intrinsics_all.append(camera.get_intrinsics())
            intrinsics_inv_all.append(camera.get_intrinsics_inv())
            c2w_all.append(camera.get_pose())
            w2c_all.append(camera.get_pose_inv())
            # get timestamps (up to temporal_dim)
            timestamps = camera.get_timestamps()  # (T,)
            timestamps = timestamps[:self.temporal_dim]  # (T_s,)
            timestamps_all.append(timestamps)

        self.intrinsics_all = np.stack(intrinsics_all)  # (N, 3, 3)
        self.intrinsics_inv_all = np.stack(intrinsics_inv_all)  # (N, 3, 3)
        self.c2w_all = np.stack(c2w_all)  # (N, 4, 4)
        self.w2c_all = np.stack(w2c_all)  # (N, 4, 4)
        self.timestamps_all = np.stack(timestamps_all)  # (N, T)

        # concat data and move to device
        for key, val in data.items():
            data[key] = np.stack(val)  # (N, T, H, W, C)
        self.data = data

        print_success(self)

    def __len__(self):
        # returns the number of cameras frames in the split
        if self.index_pixels:
            return self.nr_cameras * self.temporal_dim * self.width * self.height
        else:
            return self.nr_cameras * self.temporal_dim

    def __getitem__(self, idx) -> Camera:
        """_summary_

        Args:
            idx (int): sample index

        Returns:
            cameras_idxs (torch.Tensor, long): (1,)
            frames_idxs (torch.Tensor, long): (1,)
            intrinsics (torch.Tensor, float32): (3, 3)
            intrinsics_inv (torch.Tensor, float32): (3, 3)
            c2w (torch.Tensor, float32): (4, 4)
            w2c (torch.Tensor, float32): (4, 4)
            timestamps (torch.Tensor, float32): (1,)
            pixels (torch.Tensor, long): (H, W, 2) or (2,) [0, W-1], [0, H-1]
            rgbs (torch.Tensor, uint8): (H, W, C) or (C,)
            ...
        """
        if self.index_pixels:
            # index indexes all pixels of cameras frames in the split [0, N_cam * T * W * H]
            cam_idx = idx // (self.temporal_dim * self.width * self.height)
            frame_idx = (idx // (self.width * self.height)) % self.temporal_dim
            pixel_idx = idx % (self.width * self.height)
        else:
            # index indexes all cameras frames in the split [0, N_cam * T]
            # get camera id by dividing by the temporal dimension
            cam_idx = idx // self.temporal_dim
            frame_idx = idx % self.temporal_dim

        data = {
            "cameras_idxs": torch.tensor(cam_idx, dtype=torch.long),  # (1,)
            "frames_idxs": torch.tensor(frame_idx, dtype=torch.long),  # (1,)
            "intrinsics": torch.from_numpy(
                self.intrinsics_all[cam_idx]
            ).float(),  # (3, 3)
            "intrinsics_inv": torch.from_numpy(
                self.intrinsics_inv_all[cam_idx]
            ).float(),  # (3, 3)
            "c2w": torch.from_numpy(self.c2w_all[cam_idx]).float(),  # (4, 4)
            "w2c": torch.from_numpy(self.w2c_all[cam_idx]).float(),  # (4, 4)
            "timestamps": torch.tensor(
                self.timestamps_all[cam_idx, frame_idx]
            ).float(),  # (1,)
        }

        if self.index_pixels:
            i = pixel_idx % self.width  # x coordinate (width)
            j = pixel_idx // self.width  # y coordinate (height)
            for k, v in self.data.items():
                # keep dtype consistent with the one in Camera.data
                data[k] = torch.from_numpy(v[cam_idx, frame_idx, j, i])  # (C,)
            data["pixels"] = torch.tensor([i, j], dtype=torch.long)  # (2,)
        else:
            for k, v in self.data.items():
                # keep dtype consistent with the one in Camera.data
                data[k] = torch.from_numpy(v[cam_idx, frame_idx])  # (H, W, C)
            data["pixels"] = get_pixels(
                width=self.width, height=self.height
            )  # (H, W, 2)
            # convert pixels to long
            data["pixels"] = data["pixels"].long()
        return data

    def __str__(self) -> str:
        return f"DataSplit with {len(self)} indexable items (nr_cameras: {self.nr_cameras}, temporal_dim: {self.temporal_dim}, width: {self.width}, height: {self.height}), totalling {bytes_to_gb(self.get_memory_footprint())} GB."

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
