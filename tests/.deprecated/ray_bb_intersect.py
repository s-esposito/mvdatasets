import numpy as np
import PIL
import os
import sys
import time
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import imageio

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.utils.plotting import plot_bounding_boxes
from mvdatasets.utils.profiler import Profiler
from mvdatasets.mvdataset import MVDataset
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.utils.tensor_reel import TensorReel
from mvdatasets.utils.virtual_cameras import sample_cameras_on_hemisphere
from mvdatasets.utils.bounding_box import BoundingBox
from mvdatasets.utils.geometry import deg2rad, rot_x_3d, rot_y_3d, rot_z_3d
from mvdatasets.utils.raycasting import get_camera_rays


if __name__ == "__main__":

    # Set a random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed(seed)  # Set a random seed for GPU
    else:
        device = "cpu"
    torch.set_default_device(device)

    # Set default tensor type
    torch.set_default_dtype(torch.float32)

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code
    
    # Set datasets path
    datasets_path = "/home/stefano/Data"

    # Get dataset test preset
    dataset_name = "dmsr"
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)
    
    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True
    )
    
    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        point_cloud = np.empty((0, 3))
    
    camera = mv_data["test"][0]
    width = camera.width
    height = camera.height
    # width = 800
    # height = 800
    # vfov = 90.0
    # focal = (height / 2) / np.tan(np.deg2rad(vfov / 2))
    # cx = width / 2
    # cy = height / 2
    # intrinsics = np.array([
    #     [focal, 0, cx],
    #     [0, focal, cy],
    #     [0, 0, 1]
    # ], dtype=np.float32)
    # camera_radius = 0.5
    
    # sampled_cameras = sample_cameras_on_hemisphere(
    #     intrinsics=intrinsics,
    #     width=width,
    #     height=height,
    #     radius=camera_radius,
    #     nr_cameras=1
    # )
    # camera = sampled_cameras[0]

    # create bounding boxes
    bounding_boxes = []
    
    # create bounding boxes
    bounding_boxes = []
    scene_path = f"tests/assets/assetsnerf/{dataset_name}/{scene_name}"
    icp_path = os.path.join(scene_path, "icp")
    # bounding_boxes_path = os.path.join(scene_path, "bounding_boxes")
    
    # list files in icp directory
    scene_scale = mv_data.scene_scale_mult
    assets_names = [f.split(".")[0] for f in os.listdir(icp_path) if f.endswith(".npy")]
    for asset_name in assets_names:
        
        # load asset instances info
        asset_meta_path = os.path.join(icp_path, f"{asset_name}.npy")
        asset_meta = np.load(asset_meta_path, allow_pickle=True).item()[asset_name]
        print(f"loaded {asset_name} instances info")
        
        father_bb = asset_meta['bb']
        father_bb["dimensions"] *= scene_scale
        father_bb_pose = np.eye(4)
        father_bb_pose[:3, 3] = father_bb["center"] * scene_scale
        print(f"father_bb_pose: {father_bb_pose}")
        father_bb = BoundingBox(
            pose=father_bb_pose,
            label="P",
            color="red",
            local_scale=father_bb['dimensions'],
            device="cuda"
        )
        # father_bb.save_as_ply(bounding_boxes_path, f'bb_{asset_name}_principal')
        bounding_boxes.append(father_bb)
        
        # remove "bb" key from asset_meta
        del asset_meta["bb"]
        
        for instance_id, instance_meta in asset_meta.items():
        
            transformation_matrix = instance_meta["transformation"]
            bb_pose = transformation_matrix
            bb_pose[:3, 3] = bb_pose[:3, 3] * scene_scale
            bb = BoundingBox(
                pose=bb_pose,
                father_bb=father_bb,
                label=str(instance_id),
                color="blue",
                device="cuda"
            )
            # bb.save_as_ply(bounding_boxes_path, f'bb_{asset_name}_{instance_id}')
            bounding_boxes.append(bb)
    # finished loading bounding boxes
    
    # shoot rays from camera and intersect with boxes
    rays_o, rays_d, points_2d = get_camera_rays(camera, device=device)
    
    intersections = []
    for i, bb in enumerate(bounding_boxes):
        is_hit, t_near, t_far, p_near, p_far = bb.intersect(rays_o, rays_d)
        intersections.append([is_hit, t_near, t_far, p_near, p_far])
        print(f"is {i} bb hit?", np.any(is_hit.cpu().numpy()))
    
    points_near = []
    points_far = []
    for (is_hit, _, _, p_near, p_far) in intersections:
        points_near.append(p_near[is_hit].cpu().numpy())
        points_far.append(p_far[is_hit].cpu().numpy())
    points_near = np.concatenate(points_near, axis=0)
    points_far = np.concatenate(points_far, axis=0)
    
    near_depth = np.ones((height, width)).flatten() * np.inf
    far_depth = np.ones((height, width)).flatten() * np.inf
    for (is_hit, t_near, t_far, _, _) in intersections:
        # t_near
        t_near_np = t_near.cpu().numpy()
        updates = t_near_np < near_depth
        updates *= is_hit.cpu().numpy()
        near_depth[updates] = t_near_np[updates]
        # t_far
        t_far_np = t_far.cpu().numpy()
        updates = t_far_np < far_depth
        updates *= is_hit.cpu().numpy()
        far_depth[updates] = t_far_np[updates]
    
    near_depth = near_depth.reshape(height, width)
    far_depth = far_depth.reshape(height, width)
    
    plt.imshow(near_depth, cmap="jet")
    plt.colorbar()
    plt.savefig(
        os.path.join("plots", "ray_bb_hit_near_depth.png"),
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()
    
    plt.imshow(far_depth, cmap="jet")
    plt.colorbar()
    plt.savefig(
        os.path.join("plots", "ray_bb_hit_far_depth.png"),
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()
    
    fig = plot_bounding_boxes(
        bounding_boxes=bounding_boxes,
        point_clouds=[point_cloud, points_near, points_far],
        cameras=[camera],
        azimuth_deg=230,
        elevation_deg=60,
        scene_radius=1.0,
        max_nr_points=1000,
        up="z",
        figsize=(15, 15),
        draw_origin=False,
        draw_frame=False,
        title="bounding boxes and intersection points",
    )
    # plt.show()
    
    plt.savefig(
        os.path.join("plots", f"bbs.png"),
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        dpi=300
    )
    plt.close()