import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os.path as osp


def get_tracks_3d(
    self, num_samples: int, step: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get 3D tracks from the dataset.
    From Shape-of-Motion.

    Args:
        num_samples (int | None): The number of samples to fetch. If None,
            fetch all samples. If not None, fetch roughly a same number of
            samples across each frame. Note that this might result in
            number of samples less than what is specified.
        step (int): The step to temporally subsample the track.

    Returns 3D tracks:
        coordinates (N, T, 3),
        visibles (N, T),
        invisibles (N, T),
        confidences (N, T),
        colors (N, 3)
    """

    # TODO: assert selected camera is part of the training split
    # "fetch_tracks_3d is only available for the training split.

    # ....

    def parse_tapir_track_info(occlusions, expected_dist):
        """
        return:
            valid_visible: mask of visible & confident points
            valid_invisible: mask of invisible & confident points
            confidence: clamped confidence scores (all < 0.5 -> 0)
        """
        visiblility = 1 - F.sigmoid(occlusions)
        confidence = 1 - F.sigmoid(expected_dist)
        valid_visible = visiblility * confidence > 0.5
        valid_invisible = (1 - visiblility) * confidence > 0.5
        # set all confidence < 0.5 to 0
        confidence = confidence * (valid_visible | valid_invisible).float()
        return valid_visible, valid_invisible, confidence

    def normalize_coords(coords, h, w):
        assert coords.shape[-1] == 2
        return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0

    # TODO: get raw_tracks_2d from the dataset
    raw_tracks_2d = None
    
    # Process 3D tracks.
    inv_Ks = torch.linalg.inv(self.Ks)[::step]
    c2ws = torch.linalg.inv(self.w2cs)[::step]
    H, W = self.imgs.shape[1:3]
    filtered_tracks_3d, filtered_visibles, filtered_track_colors = [], [], []
    filtered_invisibles, filtered_confidences = [], []
    masks = self.masks * self.valid_masks * (self.depths > 0)
    masks = (masks > 0.5).float()
    for i, tracks_2d in enumerate(raw_tracks_2d):
        tracks_2d = tracks_2d.swapdims(0, 1)
        tracks_2d, occs, dists = (
            tracks_2d[..., :2],
            tracks_2d[..., 2],
            tracks_2d[..., 3],
        )
        # visibles = postprocess_occlusions(occs, dists)
        visibles, invisibles, confidences = parse_tapir_track_info(occs, dists)
        # Unproject 2D tracks to 3D.
        track_depths = F.grid_sample(
            self.depths[::step, None],
            normalize_coords(tracks_2d[..., None, :], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0]
        tracks_3d = (
            torch.einsum(
                "nij,npj->npi",
                inv_Ks,
                F.pad(tracks_2d, (0, 1), value=1.0),
            )
            * track_depths
        )
        tracks_3d = torch.einsum(
            "nij,npj->npi", c2ws, F.pad(tracks_3d, (0, 1), value=1.0)
        )[..., :3]
        # Filter out out-of-mask tracks.
        is_in_masks = (
            F.grid_sample(
                masks[::step, None],
                normalize_coords(tracks_2d[..., None, :], H, W),
                align_corners=True,
            ).squeeze()
            == 1
        )
        visibles *= is_in_masks
        invisibles *= is_in_masks
        confidences *= is_in_masks.float()
        # Get track's color from the query frame.
        track_colors = (
            F.grid_sample(
                self.imgs[i * step : i * step + 1].permute(0, 3, 1, 2),
                normalize_coords(tracks_2d[i : i + 1, None, :], H, W),
                align_corners=True,
                padding_mode="border",
            )
            .squeeze()
            .T
        )
        # at least visible 5% of the time, otherwise discard
        visible_counts = visibles.sum(0)
        valid = visible_counts >= min(
            int(0.05 * self.num_frames),
            visible_counts.float().quantile(0.1).item(),
        )

        filtered_tracks_3d.append(tracks_3d[:, valid])
        filtered_visibles.append(visibles[:, valid])
        filtered_invisibles.append(invisibles[:, valid])
        filtered_confidences.append(confidences[:, valid])
        filtered_track_colors.append(track_colors[valid])

    filtered_tracks_3d = torch.cat(filtered_tracks_3d, dim=1).swapdims(0, 1)
    filtered_visibles = torch.cat(filtered_visibles, dim=1).swapdims(0, 1)
    filtered_invisibles = torch.cat(filtered_invisibles, dim=1).swapdims(0, 1)
    filtered_confidences = torch.cat(filtered_confidences, dim=1).swapdims(0, 1)
    filtered_track_colors = torch.cat(filtered_track_colors, dim=0)

    return (
        filtered_tracks_3d,
        filtered_visibles,
        filtered_invisibles,
        filtered_confidences,
        filtered_track_colors,
    )
