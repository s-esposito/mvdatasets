import os
import numpy as np
import open3d as o3d
from rich import print
from volsurfs_py.utils.texture import RGBATexture, SHTexture


def find_mesh_files_in_path(meshes_path):
    meshes_files_found = []
    print(f"\nmeshes found in {meshes_path}")
    for file_name in os.listdir(meshes_path):
        if file_name.endswith(".obj") or file_name.endswith(".ply"):
            meshes_files_found.append(file_name)
            print(file_name)
    print("")
    return meshes_files_found


# def triangle_to_vertices_uvs_conversion(triangle_uvs, faces):
#     # find unique values and their indices
#     unique_values, unique_indices = np.unique(faces.flatten(), return_inverse=True)
#     # get the indices of the first appearance of each unique value
#     first_appearance_indices = np.array(
#         [np.where(unique_indices == i)[0][0] for i in range(len(unique_values))]
#     )
#     vertices_uvs = triangle_uvs[first_appearance_indices]
#     return vertices_uvs


def triangle_to_vertices_uvs_conversion(triangle_uvs, faces, vertices):
    flat_faces = faces.flatten()
    unique_values, unique_indices = np.unique(flat_faces, return_index=True)
    vertices_uvs = np.zeros((vertices.shape[0], 2), dtype=np.float32)
    vertices_uvs[flat_faces[unique_indices]] = triangle_uvs[unique_indices]
    return vertices_uvs


def get_isolevel_mesh_file(isolevel, meshes_files_found):
    # get mesh filename from meshes_files_found
    mesh_file_name = ""
    mesh_isolevel_ = str(float(isolevel))  # make sure it is a string with a dot
    for file_name in meshes_files_found:
        if str(float(mesh_isolevel_)) == file_name.split("_")[-1].replace(
            ".obj", ""
        ).replace(".ply", ""):
            mesh_file_name = file_name
    if mesh_file_name == "":
        raise ValueError(f"mesh for {mesh_isolevel_} not found")

    return mesh_file_name


def load_meshes_isolevels(selected_meshes_isolevels, meshes_path):
    """handles meshes loading and initialization
    args:
        selected_meshes_isolevels: list of isolevels to load (e.g. [-0.01, 0.0, 0.01])
        meshes_path: path to the meshes
    """

    # make sure the path exists
    if not os.path.exists(meshes_path):
        raise ValueError(f"mesh file {meshes_path} does not exist")

    # validate selected_meshes_isolevels
    if len(selected_meshes_isolevels) == 0:
        raise ValueError("no isolevel set in config file")
    print(f"selected_meshes_isolevels: {selected_meshes_isolevels}")

    # get list of all meshes filenames in meshes_path
    meshes_files_found = find_mesh_files_in_path(meshes_path)

    # load the mesh and add it to the scene
    meshes = []
    meshes_paths = []

    nr_meshes = len(selected_meshes_isolevels)
    for i, isolevel in enumerate(selected_meshes_isolevels):
        # load mesh
        mesh_file_name = get_isolevel_mesh_file(isolevel, meshes_files_found)
        mesh_file_path = os.path.join(meshes_path, mesh_file_name)
        print(f"loading mesh {i+1}/{nr_meshes} from file", mesh_file_path)
        mesh = Mesh(mesh_file_path, load_uvs=True)
        meshes.append(mesh)
        meshes_paths.append(mesh_file_path)
        
    return meshes, meshes_paths


class Mesh:
    def __init__(
        self,
        mesh_meta,
        load_uvs=True,
    ):  
        self.mesh_path = mesh_meta["mesh_path"]
        textures_meta = mesh_meta.get("textures", [])

        # make sure mesh_path exists
        assert os.path.exists(self.mesh_path), f"mesh path {self.mesh_path} does not exist"
        
        # load mesh
        mesh = o3d.io.read_triangle_mesh(self.mesh_path, print_progress=False)
        mesh.compute_vertex_normals()

        # vertices
        self.vertices = np.asarray(mesh.vertices).astype(np.float32)
        print(f"[INFO] mesh vertices: {self.vertices.shape}")

        # faces
        self.faces = np.asarray(mesh.triangles).astype(np.int32)
        print(f"[INFO] mesh faces: {self.faces.shape}")
        
        # normals
        self.normals = np.asarray(mesh.vertex_normals).astype(np.float32)
        print(f"[INFO] mesh normals: {self.normals.shape}")

        # uvs (if any)
        if load_uvs:
            triangle_uvs = np.asarray(mesh.triangle_uvs).astype(np.float32)
            assert triangle_uvs.shape[0] > 0, f"no uvs found in mesh {self.mesh_path}"

            self.vertices_uvs = triangle_to_vertices_uvs_conversion(
                triangle_uvs,
                self.faces,
                self.vertices,
            )

            print(
                f"[INFO] mesh vertices uvs: {self.vertices_uvs.shape if self.vertices_uvs is not None else None}"
            )
        else:
            self.vertices_uvs = None

        # load texture
        if self.vertices_uvs is not None and len(textures_meta) > 0:
            if len(textures_meta) == 1:
                self.texture = RGBATexture(textures_meta[0])
            else:
                self.texture = SHTexture(textures_meta)
        else:
            self.texture = None

        # sample a random color for each mesh face (for visualization)
        self.faces_color = np.random.rand(self.faces.shape[0], 3).astype(np.float32)
    
    @property
    def uv_idx(self):
        return self.faces
    
    @property
    def uv(self):
        return self.vertices_uvs
    
    
