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
    dataset_path = args.datasets_path
    dataset_name = args.dataset_name
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        dataset_path,
        point_clouds_paths=pc_paths,
        splits=["train", "test"],
        config=config,
        verbose=True,
    )

    if mv_data.init_sphere_radius > mv_data.scene_radius:
        print_error("init_sphere_radius > scene_radius, this can't be true")

    if len(mv_data.point_clouds) > 0:
        point_cloud = mv_data.point_clouds[0]
    else:
        point_cloud = None

    # create bounding primitives
    bounding_boxes = []
    bounding_spheres = []

    # scene
    if mv_data.scene_type == "bounded":
        bb = BoundingBox(
            pose=np.eye(4),
            local_scale=mv_data.scene_radius * 2,
            line_width=2.0,
            device=device,
        )
        bounding_boxes.append(bb)
    elif mv_data.scene_type == "unbounded":
        bs = BoundingSphere(
            pose=np.eye(4),
            local_scale=0.5,  # outer primitive
            device=device,
            verbose=True,
        )
        bounding_spheres.append(bs)

    # sdf init

    bs = BoundingSphere(
        pose=np.eye(4),
        local_scale=np.array(
            [
                mv_data.init_sphere_radius,
                mv_data.init_sphere_radius,
                mv_data.init_sphere_radius,
            ]
        ),  # outer primitive
        device=device,
        verbose=True,
    )
    bounding_spheres.append(bs)

    draw_contraction_spheres = False
    scene_type = config.get("scene_type", None)
    if scene_type == "unbounded":
        draw_contraction_spheres = True

    # Visualize cameras
    plot_3d(
        cameras=mv_data["train"],
        points_3d=[point_cloud],
        points_3d_colors=["black"],
        bounding_boxes=bounding_boxes,
        bounding_spheres=bounding_spheres,
        azimuth_deg=20,
        elevation_deg=30,
        up="z",
        scene_radius=mv_data.scene_radius,
        draw_bounding_cube=True,
        draw_image_planes=True,
        draw_cameras_frustums=False,
        draw_contraction_spheres=draw_contraction_spheres,
        figsize=(15, 15),
        title="training cameras",
        show=True,
        save_path=os.path.join("plots", f"{dataset_name}_test_cameras.png"),
    )

    # Visualize cameras
    plot_3d(
        cameras=mv_data["test"],
        points_3d=[point_cloud],
        points_3d_colors=["black"],
        bounding_boxes=bounding_boxes,
        bounding_spheres=bounding_spheres,
        azimuth_deg=20,
        elevation_deg=30,
        up="z",
        scene_radius=mv_data.scene_radius,
        draw_bounding_cube=True,
        draw_image_planes=True,
        draw_cameras_frustums=False,
        draw_contraction_spheres=draw_contraction_spheres,
        figsize=(15, 15),
        title="test cameras",
        show=True,
        save_path=os.path.join("plots", f"{dataset_name}_test_cameras.png"),
    )

    print("done")
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
