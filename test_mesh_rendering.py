import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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

camera = mv_data["test"][1]

imgs = render_mesh(camera, triangle_mesh)

plt.imshow(imgs["depth"], origin="lower")
plt.colorbar()
plt.show()
