import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.rendering import render_mesh

# Set profiler
profiler = Profiler()  # nb: might slow down the code

datasets_path = "/home/stefano/Data"
dataset_name = "dtu"
scene_name = "dtu_scan83"

# load gt mesh if exists
mesh_file_path = os.path.join("debug/meshes/", dataset_name, scene_name, "mesh.ply")

# dataset loading
mv_data = MVDataset(
    dataset_name,
    scene_name,
    datasets_path,
    point_clouds_paths=[],
    splits=["train", "test"],
    test_camera_freq=8,
    load_mask=True,
)

# load mesh

triangle_mesh = o3d.io.read_triangle_mesh(mesh_file_path)
triangle_mesh.compute_vertex_normals()

# render mesh

camera = mv_data["test"][0]

imgs = render_mesh(camera, triangle_mesh)

plt.imshow(imgs["depth"])
plt.colorbar()

#  plt.show()
plt.savefig(os.path.join("imgs", "dtu_mesh_depth.png"), dpi=300)
plt.close()