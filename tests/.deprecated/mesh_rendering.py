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
from mvdatasets.utils.images import numpy_to_image
from mvdatasets.config import get_dataset_test_preset
from mvdatasets.config import DATASETS_PATH

if __name__ == "__main__":

    dataset_name = "dtu"
    scene_name, pc_paths, config = get_dataset_test_preset(dataset_name)
    mesh_file_path = "tests/assets/meshes/dtu/dtu_scan83.ply"

    # dataset loading
    mv_data = MVDataset(
        dataset_name,
        scene_name,
        DATASETS_PATH,
        splits=["train", "test"],
    )

    # load mesh

    triangle_mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    triangle_mesh.compute_vertex_normals()

    # render mesh
    camera = mv_data["test"][2]
    imgs = render_o3d_mesh(camera, triangle_mesh)
    plt.imshow(imgs["depth"])
    plt.colorbar()
    plt.show()

    # plt.savefig(os.path.join("plots", f"{dataset_name}_mesh_depth.png"), transparent=True, dpi=300)
    # plt.close()
