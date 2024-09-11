from rich import print
import numpy as np

import torch

import cv2
from PIL import Image
from mvdatasets.utils.images import (
    uv_coords_to_pix,
    non_normalized_uv_coords_to_lerp_weights,
    non_normalize_uv_coord,
    non_normalized_uv_coords_to_interp_corners
)


class TensorTexture():
    def __init__(
        self,
        texture_np=None,
        texture_path=None,
        res=None,
        lerp=True,
        device="cuda",
    ):
        super(TensorTexture, self).__init__()
        self.device = device
        self.lerp = lerp
        
        if texture_np is None:
            if texture_path is None:
                print("[bold red]ERROR[/bold red]: texture_np and texture_path cannot be both None")
                exit()
            texture_pil = Image.open(texture_path).convert('RGBA')
            texture_np = np.array(texture_pil) / 255.0
        
        # TODO: necessary?
        texture_np = np.flipud(texture_np).copy()
        
        self.data = torch.from_numpy(texture_np).float().to(self.device)
        
        if res is not None:
            # use opencv to resize to res
            texture_cv = cv2.resize(texture_np, (res[1], res[0]), interpolation=cv2.INTER_LINEAR)
            self.data = torch.from_numpy(texture_cv).float().to(self.device)
        
        #
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], -1)
        self.shape = self.data.shape
        
        # height, width
        self.res = torch.tensor([self.shape[0], self.shape[1]]).long().to(self.device)  # [height, width]
        print("self.res", self.res)
        
        if self.lerp:
            # pad with zeros
            data_shape = self.data.shape
            # to list to modify
            data_shape = list(data_shape)
            # add 2 to first two dimensions
            data_shape[0] += 2
            data_shape[1] += 2
            padded_data = torch.zeros(data_shape, device=self.device)
            padded_data[1:-1, 1:-1] = self.data
            self.data = padded_data
        
        print("self.data.shape", self.data.shape)
        
    def __call__(self, uv_coords):
        
        if self.lerp:
            # uv coords are width, height
            # res is height, width
            # flip=True
            # results are non normalized uv coordinates
            uv_coords_nn = non_normalize_uv_coord(uv_coords, self.res, flip=True)
            
            # results are non normalized uv coordinates of the corners
            uv_corners_coords_nn = non_normalized_uv_coords_to_interp_corners(uv_coords_nn)  # [N, 4, 2]
            
            # find lerp weights
            lerp_weights = non_normalized_uv_coords_to_lerp_weights(
                uv_coords_nn,
                uv_corners_coords_nn
            )
            
            # convert to (padded) pixel coordinates
            uv_corners_pix = uv_corners_coords_nn.floor().long() + 1
            uv_corners_pix = uv_corners_pix.reshape(-1, 2)  # [N*4, 2]
            # print("uv_corners_pix.shape", uv_corners_pix.shape)
            # query texture at pixels coordinates
            vertices_values = self.data[uv_corners_pix[:, 1], uv_corners_pix[:, 0]]
            # print("vertices_values.shape", vertices_values.shape)
            vertices_values = vertices_values.reshape(uv_coords.shape[0], 4, -1)
            
            # interpolate
            # print("lerp_weights.shape", lerp_weights.shape)
            # print("vertices_values.shape", vertices_values.shape)
            output = (vertices_values * lerp_weights).sum(dim=1)
        else:
            # anchor uv_coords [0, 1] to pixel coordinates
            # uv coords are width, height
            # res is height, width
            # flip=True
            # results are pixel coordinates
            uv_pix = uv_coords_to_pix(uv_coords, self.res, flip=True)  # u, v
            
            # pixel coordinates are width, height
            # images is height, width
            # query texture at pixel coordinates
            output = self.data[uv_pix[:, 1], uv_pix[:, 0]]
        
        return output
    
    def to_numpy(self):
        return self.data[:self.shape[0], :self.shape[1]].detach().cpu().numpy()