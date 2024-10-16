import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.open3d_rendering import render_o3d_mesh
from mvdatasets.utils.images import numpy2image
from mvdatasets.utils.common import get_dataset_test_preset

if __name__ == "__main__":

    # Set profiler
    profiler = Profiler()  # nb: might slow down the code

    # Set datasets path
    datasets_path = "/home/stefano/Data"

    # Get dataset test preset
    # if len(sys.argv) > 1:
    #     dataset_name = sys.argv[1]
    # else:
    dataset_name = "dmsr"
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)

    # load gt mesh if exists
    mesh_file_path = "/home/stefano/Data/dmsr/dinning/dinning.ply"

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        datasets_path,
        splits=["train", "test"],
    )

    # load mesh

    triangle_mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    triangle_mesh.compute_vertex_normals()

    # render mesh

    splits = ["test", "train"]
    for split in splits:
        for camera in mv_data[split]:
            imgs = render_o3d_mesh(camera, triangle_mesh)
            depth = imgs["depth"]
            plt.imshow(depth)
            plt.show()
            break
        break
            # save_path = os.path.join(datasets_path, dataset_name, scene_name, split, "depth")
            # save_nr = format(camera.camera_idx, "04d")
            # np.save(os.path.join(save_path, f"d_{save_nr}"), depth)
            # depth_pil = numpy2image(depth)
            # save_path = os.path.join(datasets_path, dataset_name, scene_name, split, "depth")
            # save_nr = format(camera.camera_idx, "04d")
            # depth_pil.save(f"d_{save_nr}.png")
            
            # plt.imshow(imgs["depth"])
            # plt.colorbar()
            # plt.show()
            
    # camera = mv_data["test"][0]

    # imgs = render_o3d_mesh(camera, triangle_mesh)

    # plt.imshow(imgs["depth"])
    # plt.colorbar()

    # #  plt.show()
    # plt.savefig(os.path.join("plots", f"{dataset_name}_mesh_depth.png"), transparent=True, dpi=300)
    # plt.close()
