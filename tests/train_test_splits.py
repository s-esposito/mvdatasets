import tyro
import numpy as np
import os
from config import get_dataset_test_preset
from config import Args
from mvdatasets.visualization.matplotlib import plot_3d
from mvdatasets.mvdataset import MVDataset
from mvdatasets.geometry.primitives.bounding_box import BoundingBox
from mvdatasets.geometry.primitives.bounding_sphere import BoundingSphere
from mvdatasets.utils.printing import print_error


def main(args: Args):

    device = args.device
    datasets_path = args.datasets_path
    dataset_name = args.dataset_name
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True,
    )

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        point_cloud = None
    
    # sdf init

    bs = BoundingSphere(
        pose=np.eye(4),
        local_scale=mv_data.get_sphere_init_radius(),
        device=device,
        verbose=True,
    )

    bbs = []
    draw_bounding_cube = True
    draw_contraction_spheres = False
    scene_type = config.get("scene_type", None)
    if scene_type == "bounded":
        # scene scale bb
        bb = BoundingBox(
            pose=np.eye(4),
            local_scale=mv_data.get_foreground_radius() * 2,
            device=device,
        )
        bbs.append(bb)
    if scene_type == "unbounded":
        draw_bounding_cube = False
        draw_contraction_spheres = True

    # Visualize cameras
    plot_3d(
        cameras=mv_data["train"],
        points_3d=[point_cloud],
        points_3d_colors=["black"],
        bounding_boxes=bbs,
        bounding_spheres=[bs],
        azimuth_deg=20,
        elevation_deg=30,
        up="z",
        scene_radius=mv_data.get_scene_radius(),
        draw_bounding_cube=draw_bounding_cube,
        draw_image_planes=True,
        draw_cameras_frustums=False,
        draw_contraction_spheres=draw_contraction_spheres,
        figsize=(15, 15),
        title="training cameras",
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_train_cameras.png"),
    )

    # Visualize cameras
    plot_3d(
        cameras=mv_data["test"],
        points_3d=[point_cloud],
        points_3d_colors=["black"],
        bounding_boxes=bbs,
        bounding_spheres=[bs],
        azimuth_deg=20,
        elevation_deg=30,
        up="z",
        scene_radius=mv_data.get_scene_radius(),
        draw_bounding_cube=draw_bounding_cube,
        draw_image_planes=True,
        draw_cameras_frustums=False,
        draw_contraction_spheres=draw_contraction_spheres,
        figsize=(15, 15),
        title="test cameras",
        show=False,
        save_path=os.path.join("plots", f"{dataset_name}_test_cameras.png"),
    )

    print("done")
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
