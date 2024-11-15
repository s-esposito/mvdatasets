import torch


class TensorMesh:
    def __init__(self, mesh, device="cuda"):
        self.vertices = torch.from_numpy(mesh.vertices).to(device)
        self.faces = torch.from_numpy(mesh.faces).to(device)

        self.vertices_uvs = None
        if mesh.vertices_uvs is not None:
            self.vertices_uvs = torch.from_numpy(mesh.vertices_uvs).to(device)

        # self.texture = None
        # if mesh.texture is not None:
        #     self.texture = torch.from_numpy(mesh.texture.image).to(device)

        # for visualization only

        self.faces_color = None
        if mesh.faces_color is not None:
            self.faces_color = torch.from_numpy(mesh.faces_color).to(device)

        self.mesh_color = torch.rand(1, 3).to(device)

    def get_faces_uvs(self):
        if self.vertices_uvs is None:
            return None
        return self.vertices_uvs[self.faces.flatten()].view(-1, 3, 2)
