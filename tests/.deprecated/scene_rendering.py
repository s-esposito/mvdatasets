import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.open3d_rendering import render_o3d_scene
from mvdatasets.geometry.rigid import apply_transformation_3d
from mvdatasets.config import get_dataset_test_preset

if __name__ == "__main__":

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code

    # Get dataset test preset
    # if len(sys.argv) > 1:
    #     dataset_name = sys.argv[1]
    # else:
    dataset_name = "dmsr"
    scene_name, pc_paths, config = get_dataset_test_preset(args.dataset_name)

    # dataset loading
    mv_data = MVDataset(
        args.dataset_name,
        scene_name,
        args.datasets_path,
        splits=["train", "test"],
        config=config,
        verbose=True,
    )

    # scene path (folder containing only mesh files)
    scene_dir = "/home/stefano/Data/dmsr/dinning/meshes"

    # get all files in scene dir
    files = os.listdir(scene_dir)
    nr_objects = len(files)
    print(files)

    # setup scene
    o3d_scene = o3d.t.geometry.RaycastingScene()

    # load meshes and add to scene
    rotation = mv_data.global_transform[:3, :3]
    translation = mv_data.global_transform[:3, 3]
    for mesh_file in files:
        o3d_mesh = o3d.io.read_triangle_mesh(os.path.join(scene_dir, mesh_file))
        o3d_mesh_vertices = np.asarray(o3d_mesh.vertices)
        o3d_mesh_vertices = apply_transformation_3d(
            o3d_mesh_vertices, mv_data.global_transform
        )
        o3d_mesh.vertices = o3d.utility.Vector3dVector(o3d_mesh_vertices)
        o3d_mesh.compute_vertex_normals()
        o3d_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))

    # render mesh
    splits = ["test", "train"]
    for split in splits:
        save_path = os.path.join(args.datasets_path, dataset_name, scene_name, split)
        for camera in mv_data.get_split(split):
            # print(camera.camera_idx)
            imgs = render_o3d_scene(camera, o3d_scene)
            geom_ids = imgs["geom_ids"]
            depth = imgs["depth"]
            print("min depth", np.min(depth))
            print("max depth", np.max(depth))
            # print(geom_ids.shape)
            # print(np.unique(geom_ids))
            # plt.imshow(geom_ids, vmax=nr_objects)
            plt.imshow(depth, cmap="jet")
            plt.colorbar()
            plt.show()
            break
            # save_nr = format(camera.camera_idx, "04d")
            # os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
            # os.makedirs(os.path.join(save_path, "instance_mask"), exist_ok=True)
            # np.save(os.path.join(save_path, "depth", f"d_{save_nr}"), depth)
            # np.save(os.path.join(save_path, "instance_mask", f"instance_mask_{save_nr}"), geom_ids)
