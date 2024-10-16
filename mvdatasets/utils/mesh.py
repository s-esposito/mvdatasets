import os
import numpy as np
import open3d as o3d
from rich import print
from mvdatasets.utils.texture import RGBATexture, SHTexture


def triangle_to_vertices_uvs_conversion(triangle_uvs, faces, vertices):
    flat_faces = faces.flatten()
    unique_values, unique_indices = np.unique(flat_faces, return_index=True)
    vertices_uvs = np.zeros((vertices.shape[0], 2), dtype=np.float32)
    vertices_uvs[flat_faces[unique_indices]] = triangle_uvs[unique_indices]
    return vertices_uvs


class Mesh:
    def __init__(
        self,
        o3d_mesh=None,
        mesh_meta=None,
        verbose=True,
    ):
        # only one of o3d_mesh and mesh_meta should be provided
        if o3d_mesh is not None and mesh_meta is not None:
            print("[bold red]ERROR[/bold red]: only one of o3d_mesh and mesh_meta should be provided")
            exit(1)
            
        # at least one of o3d_mesh and mesh_meta should be provided
        if o3d_mesh is None and mesh_meta is None:
            print("[bold red]ERROR[/bold red]: at least one of o3d_mesh and mesh_meta should be provided")
            exit(1)
            
        if o3d_mesh is not None:
            
            mesh = o3d_mesh
            textures_meta = []
        
        else:
        
            self.mesh_path = mesh_meta["mesh_path"]
            textures_meta = mesh_meta.get("textures", [])

            # make sure mesh_path exists
            assert os.path.exists(self.mesh_path), f"mesh path {self.mesh_path} does not exist"
        
            # load mesh
            mesh = o3d.io.read_triangle_mesh(self.mesh_path, print_progress=False)
        
        mesh.compute_vertex_normals()

        # vertices
        self.vertices = np.asarray(mesh.vertices).astype(np.float32)
        if verbose:
            print(f"[bold blue]INFO[/bold blue]: mesh vertices: {self.vertices.shape}")

        # faces
        self.faces = np.asarray(mesh.triangles).astype(np.int32)
        if verbose:
            print(f"[bold blue]INFO[/bold blue]: mesh faces: {self.faces.shape}")
        
        # normals
        self.normals = np.asarray(mesh.vertex_normals).astype(np.float32)
        if verbose:
            print(f"[bold blue]INFO[/bold blue]: mesh normals: {self.normals.shape}")

        # uvs (if any)
        triangle_uvs = np.asarray(mesh.triangle_uvs).astype(np.float32)
        if triangle_uvs.shape[0] == 0:
            self.vertices_uvs = None
            self.has_uvs = False
        else:
            self.vertices_uvs = triangle_to_vertices_uvs_conversion(
                triangle_uvs,
                self.faces,
                self.vertices,
            )
            self.has_uvs = True
        if verbose:
            print(
                f"[bold blue]INFO[/bold blue]: mesh vertices uvs: {self.vertices_uvs.shape if self.vertices_uvs is not None else None}"
            )

        # load texture
        if len(textures_meta) > 0:
            assert self.has_uvs, "mesh must have uvs to load texture"
            # TODO: support RGBATexture
            # if len(textures_meta) == 1:
            #     print("[bold blue]INFO[/bold blue]: loading RGBATexture")
            #     self.texture = RGBATexture(textures_meta[0], verbose=verbose)
            # else:
            print("[bold blue]INFO[/bold blue]: loading SHTexture")
            self.texture = SHTexture(textures_meta, verbose=verbose)
            self.has_texture = True
        else:
            self.texture = None
            self.has_texture = False

        # sample a random color for each mesh face (for visualization)
        self.faces_color = np.random.rand(self.faces.shape[0], 3).astype(np.float32)
    
    @property
    def uv_idx(self):
        return self.faces
    
    @property
    def uv(self):
        return self.vertices_uvs
    
    
