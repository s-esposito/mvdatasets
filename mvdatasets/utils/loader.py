import torch
from torch.utils.data import Sampler


class DatasetSampler(Sampler):
    """Custom sampler for the dataset.
    Args:
        dataset (Dataset): dataset to sample from
        shuffle (bool): shuffle the data or not
    """

    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.nr_samples = len(dataset)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(self.nr_samples))
        else:
            return iter(torch.arange(self.nr_samples))

    def __len__(self):
        return self.nr_samples


def custom_collate(batch):
    idx, rays_o, rays_d, gt_rgb, gt_mask, frame_idx = zip(*batch)
    return (
        torch.stack(idx),
        torch.stack(rays_o),
        torch.stack(rays_d),
        torch.stack(gt_rgb),
        torch.stack(gt_mask),
        torch.stack(frame_idx),
    )


def get_next_batch(data_loader):
    """calls next on the data_loader and returns a batch of data"""
    idx, rays_o, rays_d, gt_rgb, gt_mask, frame_idx = next(iter(data_loader))
    idx = idx.repeat_interleave(data_loader.dataset.per_camera_rays_batch_size)
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    gt_rgb = gt_rgb.view(-1, 3)
    gt_mask = gt_mask.view(-1, 1)
    frame_idx = frame_idx.repeat_interleave(
        data_loader.dataset.per_camera_rays_batch_size
    )

    # print("idx", idx.shape, idx.device)
    # print("rays_o", rays_o.shape, rays_o.device)
    # print("rays_d", rays_d.shape, rays_d.device)
    # print("gt_rgb", gt_rgb.shape, gt_rgb.device)
    # print("gt_mask", gt_mask.shape, gt_mask.device)
    # print("frame_idx", frame_idx.shape, frame_idx.device)

    if data_loader.sampler.shuffle:
        rand_perm = torch.randperm(idx.shape[0], device=rays_o.device)
        idx = idx[rand_perm]
        rays_o = rays_o[rand_perm]
        rays_d = rays_d[rand_perm]
        gt_rgb = gt_rgb[rand_perm]
        gt_mask = gt_mask[rand_perm]
        frame_idx = frame_idx[rand_perm]

    return idx, rays_o, rays_d, gt_rgb, gt_mask, frame_idx
