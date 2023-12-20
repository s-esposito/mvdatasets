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
from mvdatasets.utils.images import numpy2image

# Set profiler
profiler = Profiler()  # nb: might slow down the code

# Set datasets path
datasets_path = "/home/stefano/Data"

# # test DTU
# dataset_name = "dtu"
# scene_name = "dtu_scan83"
# pc_path = "debug/meshes/dtu/dtu_scan83.ply"
# config = {}

# # test blender
# dataset_name = "blender"
# scene_name = "lego"
# pc_path = "debug/point_clouds/blender/lego.ply"
# config = {}

# # test blendernerf
# dataset_name = "blendernerf"
# scene_name = "plushy"
# pc_path = "debug/meshes/blendernerf/plushy.ply"
# config = {
#     "load_mask": 1,
#     "scene_scale_mult": 0.4,
#     "rotate_scene_x_axis_deg": -90,
#     "sphere_radius": 0.6,
#     "white_bg": 1,
#     "test_skip": 10,
#     "subsample_factor": 1.0
# }

# test dmsr
dataset_name = "dmsr"
scene_name = "dinning"
config = {}

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
        imgs = render_mesh(camera, triangle_mesh)
        depth = imgs["depth"]
        save_path = os.path.join(datasets_path, dataset_name, scene_name, split, "depth")
        save_nr = format(camera.camera_idx, "04d")
        np.save(os.path.join(save_path, f"d_{save_nr}"), depth)
        # depth_pil = numpy2image(depth)
        # save_path = os.path.join(datasets_path, dataset_name, scene_name, split, "depth")
        # save_nr = format(camera.camera_idx, "04d")
        # depth_pil.save(f"d_{save_nr}.png")
        
        # plt.imshow(imgs["depth"])
        # plt.colorbar()
        # plt.show()
        
# camera = mv_data["test"][0]

# imgs = render_mesh(camera, triangle_mesh)

# plt.imshow(imgs["depth"])
# plt.colorbar()

# #  plt.show()
# plt.savefig(os.path.join("imgs", f"{dataset_name}_mesh_depth.png"), transparent=True, dpi=300)
# plt.close()
