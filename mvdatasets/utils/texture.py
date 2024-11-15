import numpy as np
import torch
import math
from rich import print
from PIL import Image

from mvdatasets.utils.images import image_to_numpy


def sample_texture(image, uvs):
    """
    Args:
        image: torch.tensor, [H, W, C] or [H, W, C, F]
        uvs: torch.tensor, [N, 2] in [0, 1]
    Out:
        vals: torch.tensor, [N, C] or [N, C, F]
    """
    # get image dims
    height, width = image.shape[:2]
    assert height == width, "only square textures are supported"
    texture_res = height
    # convert to uv coordinates
    uvs = uvs * torch.tensor([texture_res, texture_res], device=uvs.device)
    # convert to pixel coordinates
    uvs[:, 0] = torch.clamp(uvs[:, 0], 0, texture_res - 1)
    uvs[:, 1] = torch.clamp(uvs[:, 1], 0, texture_res - 1)
    uvs = uvs.int()
    # sample
    vals = image[uvs[:, 0], uvs[:, 1]]
    return vals


class RGBATexture:
    def __init__(
        self,
        texture_meta: dict,
        verbose=True,
    ):

        texture_path = texture_meta["texture_path"]
        # ignore texture scale as values should always be in [0, 1]
        assert texture_path.endswith(
            ".png"
        ), f"texture {texture_path} must be a .png file"
        if verbose:
            print(
                f"[bold blue]INFO[/bold blue]: loading texture {texture_path.split('/')[-1]}"
            )
        texture = image_to_numpy(Image.open(texture_path)).astype(np.float32)

        if "texture_scale" not in texture_meta:
            texture_meta["texture_scale"] = [0, 1]
        texture_scale = texture_meta["texture_scale"]
        min_val = texture_scale[0]
        max_val = texture_scale[1]
        scale = max_val - min_val
        texture = texture * scale + min_val

        if verbose:
            print(
                f"[bold blue]INFO[/bold blue]: texture {texture_path.split('/')[-1]} loaded: {texture.shape}"
            )
        self.image = texture
        self.height, self.width, self.nr_channels = self.image.shape


class SHTexture:
    def __init__(
        self,
        textures_meta: list,
        verbose=True,
    ):
        sh_deg = math.sqrt(len(textures_meta)) - 1
        assert sh_deg.is_integer(), "number of textures must be a square number"
        sh_deg = int(sh_deg)
        print(f"[bold blue]INFO[/bold blue]: sh_deg {sh_deg}")

        all_sh_coeffs = []
        channel_sh_coeffs = []
        nr_sh_coeffs = (sh_deg + 1) ** 2
        nr_read_sh_coeffs = 0

        for texture_meta in textures_meta:

            texture_path = texture_meta["texture_path"]

            if "texture_scale" not in texture_meta:
                texture_meta["texture_scale"] = [0, 1]
            else:
                texture_scale = texture_meta["texture_scale"]
            min_val = texture_scale[0]
            max_val = texture_scale[1]

            assert texture_path.endswith(
                ".png"
            ), f"texture {texture_path} must be a .png file"
            if verbose:
                print(
                    f"[bold blue]INFO[/bold blue]: loading texture {texture_path.split('/')[-1]}"
                )
            sh_coeffs = image_to_numpy(Image.open(texture_path)).astype(np.float32)

            sh_coeffs = sh_coeffs * (max_val - min_val) + min_val

            channel_sh_coeffs.append(sh_coeffs)

            nr_read_sh_coeffs += 4
            if verbose:
                print(
                    f"[bold blue]INFO[/bold blue]: texture {texture_path.split('/')[-1]} loaded: {sh_coeffs.shape}"
                )

            if sh_deg == 0:
                all_sh_coeffs.append(sh_coeffs)
                break

            if nr_read_sh_coeffs == nr_sh_coeffs:
                image = np.concatenate(channel_sh_coeffs, axis=-1)
                all_sh_coeffs.append(image)
                channel_sh_coeffs = []
                nr_read_sh_coeffs = 0

        if len(all_sh_coeffs) > 1:
            texture = np.stack(all_sh_coeffs, axis=2)
        else:
            texture = all_sh_coeffs[0]
            # unsqueeze last dim
            texture = np.expand_dims(texture, axis=-1)

        if verbose:
            print(
                f"[bold blue]INFO[/bold blue]: sh coeffs params {texture.shape} loaded"
            )
        self.image = texture
        self.height, self.width, self.nr_channels, _ = self.image.shape
