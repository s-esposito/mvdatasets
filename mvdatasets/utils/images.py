from PIL import Image
import numpy as np
import torch
import os
import cv2


def bilinear_downscale(img_np, times=1):
    # Resize the image using INTER_LINEAR interpolation
    for _ in range(times):
        # Get the dimensions of the input image
        height, width = img_np.shape[:2]
        img_np = cv2.resize(img_np, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
    return img_np


def bilinear_upscale(img_np, times=1):
    # Resize the image using INTER_LINEAR interpolation
    for _ in range(times):
        # Get the dimensions of the input image
        height, width = img_np.shape[:2]
        img_np = cv2.resize(img_np, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
    return img_np


def get_pixel_corners(uv_pix_nn):
    """returns pix corners in non-normalised uv space"""
    uv_coords_0 = uv_pix_nn  # 0, 0
    uv_coords_1 = uv_pix_nn + torch.tensor([1, 0], device=uv_pix_nn.device)  # 0, 1
    uv_coords_2 = uv_pix_nn + torch.tensor([0, 1], device=uv_pix_nn.device)  # 1, 0
    uv_coords_3 = uv_pix_nn + torch.tensor([1, 1], device=uv_pix_nn.device)  # 1, 1
    uv_interp_corners_nn = torch.stack([uv_coords_0, uv_coords_1, uv_coords_2, uv_coords_3], dim=1)
    return uv_interp_corners_nn


def normalize_uv_coord(uv_coords, res):
    uv_coords = uv_coords / torch.flip(res, dims=[0])
    return uv_coords


def non_normalize_uv_coord(uv_coords, res):
    uv_coords = uv_coords * torch.flip(res, dims=[0])
    return uv_coords


def uv_coord_to_pix(uv_coord, res):
    # convert uv to pixel coordinates
    uv_pix = non_normalize_uv_coord(uv_coord, res).floor().long()
    return uv_pix


def non_normalized_uv_coord_to_interp_corners(uv_coord_nn):
    # print("uv_coord_nn", uv_coord_nn)
    # shifted space (where center of pixel is at upper left corner of each texel)
    uv_coord_shifted = uv_coord_nn - 0.5
    # print("uv_coord_shifted", uv_coord_shifted)
    uv_pix_shifted = uv_coord_shifted.floor()
    # print("uv_pix_shifted", uv_pix_shifted)
    uv_corners_coords_shifted = get_pixel_corners(uv_pix_shifted)
    # print(uv_corners_coords_shifted)
    uv_corners_coords_nn = uv_corners_coords_shifted + 0.5
    # print(uv_corners_coords)
    return uv_corners_coords_nn


def pix_to_texel_center_uv_coord(uv_pix, res):
    # convert pixel coordinates to uv normalized coordinates of the texel center
    uv_coords = (uv_pix + 0.5) / torch.flip(res, dims=[0])
    return uv_coords


def non_normalized_uv_coord_to_lerp_weights(uv_coords_nn, uv_corners_coords_nn):
    diff = uv_coords_nn - uv_corners_coords_nn[:, 0, :]
    lerp_weights = torch.zeros((uv_coords_nn.shape[0], 4), device=uv_coords_nn.device)
    # 0, 0
    lerp_weights[:, 0] = (1.0 - diff[:, 1]) * (1.0 - diff[:, 0])
    # 0, 1
    lerp_weights[:, 1] = (1.0 - diff[:, 1]) * diff[:, 0]
    # 1, 0
    lerp_weights[:, 2] = diff[:, 1] * (1.0 - diff[:, 0])
    # 1, 1
    lerp_weights[:, 3] = diff[:, 1] * diff[:, 0]
    return lerp_weights.unsqueeze(-1)


def image_uint8_to_float32(tensor):
    """converts uint8 tensor to float32"""    
    if torch.is_tensor(tensor):
        if tensor.dtype == torch.float32:
            return tensor
        return tensor.type(torch.float32) / 255.0
    elif isinstance(tensor, np.ndarray):
        if tensor.dtype == np.float32:
            return tensor
        return tensor.astype(np.float32) / 255.0
    raise ValueError("tensor must be torch.tensor or np.ndarray")


def image_float32_to_uint8(tensor):
    """converts float32 tensor to uint8"""
    if torch.is_tensor(tensor):
        if tensor.dtype == torch.uint8:
            return tensor
        return (tensor * 255).type(torch.uint8)
    elif isinstance(tensor, np.ndarray):
        if tensor.dtype == np.uint8:
            return tensor
        return (tensor * 255).astype(np.uint8)
    raise ValueError("tensor must be torch.tensor or np.ndarray")


def image_to_numpy(pil_image, use_lower_left_origin=False, use_uint8=False):
    """
    Convert a PIL Image to a numpy array.
    If use_uint8 is False, the values are in [0, 1] and the dtype is float32.
    If use_uint8 is True, the values are in [0, 255] and the dtype is uint8.
    """
    if use_lower_left_origin:
        # flip vertically
        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    if use_uint8:
        # use int8
        img = np.array(pil_image, dtype=np.uint8)
    else:
        # (default) use float32
        img = np.array(pil_image, dtype=np.float32) / 255.0
    return img


def numpy_to_image(np_image, use_lower_left_origin=False):
    """Convert a numpy array to a PIL Image with int values in 0 and 255."""
    # if grayscale, repeat 3 times
    if np_image.shape[-1] == 1:
        # concat 3 times
        np_image_ = np.concatenate([np_image, np_image, np_image], axis=-1)
    else:
        np_image_ = np_image
    # convert to PIL image
    pil_image = Image.fromarray(np.uint8(np_image_ * 255))
    if use_lower_left_origin:
        # flip vertically
        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    return pil_image


def tensor_to_image(torch_tensor):
    """Convert a torch tensor to a PIL Image with int values in 0 and 255."""
    np_array = torch_tensor.cpu().numpy()
    return numpy_to_image(np_array)


def image_to_tensor(pil_image, device="cpu"):
    """Convert a PIL Image to a torch tensor with values in 0 and 1."""
    np_array = image_to_numpy(pil_image)
    return torch.from_numpy(np_array).float().to(device)


def flip_image_vertically(image):
    """Flip a PIL Image vertically."""
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def flip_image_horizontally(image):
    """Flip a PIL Image horizontally."""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def save_numpy_as_png(img_np, save_dir_path, img_filename, append_format=True):
    # create path is not exists
    os.makedirs(save_dir_path, exist_ok=True)
    # convert to Image and save
    img_np_ = np.round(np.clip(img_np, 0, 1) * 255)  # [0, 1] -> [0, 255]
    img_pil = Image.fromarray(np.uint8(img_np_))
    # write PIL Image to file
    save_path = os.path.join(save_dir_path, img_filename)
    if append_format:
        save_path += ".png"
    img_pil.save(save_path)