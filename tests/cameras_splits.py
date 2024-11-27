import tyro
import numpy as np
import os
from config import get_dataset_test_preset
from config import Args
from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.utils.printing import print_error, print_warning


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
        verbose=True,
    )

    # # sdf init
    # bs = BoundingSphere(
    #     pose=np.eye(4),
    #     local_scale=mv_data.get_sphere_init_radius(),
    #     device=device,
    #     verbose=True,
    # )
    
    # foreground bb
    bb = BoundingBox(
        pose=np.eye(4),
        local_scale=mv_data.get_foreground_radius() * 2,
        device=device,
    )
    
    # scene type
    scene_type = config.get("scene_type", None)
    if scene_type == "bounded":
        draw_bounding_cube = True
        draw_contraction_spheres = False
        
    if scene_type == "unbounded":
        draw_bounding_cube = False
        draw_contraction_spheres = True
        
        if mv_data.get_scene_radius() > 1.0:
            print_warning("scene radius is greater than 1.0, contraction spheres will not be displayed")
            bb = None
            draw_contraction_spheres = False

    # Visualize cameras
    for split in mv_data.get_splits():
        
        nr_cameras = len(mv_data.get_split(split))
        if nr_cameras // 50 > 1:
            draw_every_n_cameras = nr_cameras // 50
            print_warning(f"{split} has too many cameras; displaying one every {draw_every_n_cameras}")
        else:
            draw_every_n_cameras = 1
            
        plot_3d(
            cameras=mv_data.get_split(split),
            draw_every_n_cameras=draw_every_n_cameras,
            point_clouds=mv_data.point_clouds,
            bounding_boxes=[bb] if bb is not None else [],
            # bounding_spheres=[bs],
            azimuth_deg=20,
            elevation_deg=30,
            up="z",
            scene_radius=mv_data.get_scene_radius(),
            draw_bounding_cube=draw_bounding_cube,
            draw_image_planes=True,
            draw_cameras_frustums=False,
            draw_contraction_spheres=draw_contraction_spheres,
            figsize=(15, 15),
            title=f"{split} cameras",
            show=True,
            save_path=os.path.join("plots", f"{dataset_name}_{split}_cameras.png"),
        )

    print("done")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
