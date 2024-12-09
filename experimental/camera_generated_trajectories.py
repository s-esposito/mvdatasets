import tyro
import numpy as np
import os
from config import get_dataset_test_preset
from config import Args
from mvdatasets.visualization.matplotlib import plot_camera_trajectory
from mvdatasets.visualization.video_gen import make_video_camera_trajectory
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.printing import print_error, print_warning
from mvdatasets.geometry.trajectories import (
    generate_spiral_path,
    generate_ellipse_path_z,
    generate_ellipse_path_y,
    generate_interpolated_path,
)
from config import Args

# TODO: write tests for the functions in mvdatasets.geometry.trajectories

#     c2w_all = torch.from_numpy(c2w_all).float().to(device)
#     K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
#     width, height = list(self.parser.imsize_dict.values())[0]

#     # save to video
#     video_dir = f"{cfg.result_dir}/videos"
#     os.makedirs(video_dir, exist_ok=True)
#     writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
#     for i in tqdm.trange(len(c2w_all), desc="Rendering trajectory"):
#         camtoworlds = c2w_all[i : i + 1]
#         Ks = K[None]

#         renders, _, _ = self.rasterize_splats(
#             camtoworlds=camtoworlds,
#             Ks=Ks,
#             width=width,
#             height=height,
#             sh_degree=cfg.sh_degree,
#             near_plane=cfg.near_plane,
#             far_plane=cfg.far_plane,
#             render_mode="RGB+ED",
#         )  # [1, H, W, 4]
#         colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
#         depths = renders[..., 3:4]  # [1, H, W, 1]
#         depths = (depths - depths.min()) / (depths.max() - depths.min())
#         canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

#         # write images
#         canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
#         canvas = (canvas * 255).astype(np.uint8)
#         writer.append_data(canvas)
#     writer.close()
#     print(f"Video saved to {video_dir}/traj_{step}.mp4")


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    test_preset = get_dataset_test_preset(dataset_name)
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
        verbose=True,
    )

    # selected trajectory
    traj_path = "interp"

    # get all camera poses in split
    c2w_all = mv_data.get_all_split_poses("train")
    # scene_radius = mv_data.get_scene_radius()
    # bounds = None  # TODO: get bounds from dataset
    # spiral_radius_scale = 1.0

    # new c2w_all are all [N, 4, 4]
    if traj_path == "interp":
        c2w_all = generate_interpolated_path(poses=c2w_all, n_interp=1)

    elif traj_path == "ellipse":
        height = c2w_all[:, 2, 3].mean()
        c2w_all = generate_ellipse_path_z(c2w_all, height=height)

    # elif traj_path == "spiral":
    #     c2w_all = generate_spiral_path(
    #         c2w_all,
    #         bounds=bounds * scene_radius,
    #         spiral_scale_r=spiral_radius_scale,
    #     )

    else:
        raise ValueError(f"Render trajectory type not supported: {traj_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
