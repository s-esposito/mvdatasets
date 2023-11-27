from PIL import Image
import numpy as np
import torch


def image2numpy(pil_image, use_lower_left_origin=False):
    """Convert a PIL Image to a numpy array with values in 0 and 1."""
    if use_lower_left_origin:
        # flip vertically
        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(pil_image) / 255


def numpy2image(np_image, use_lower_left_origin=False):
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


def tensor2image(torch_tensor):
    """Convert a torch tensor to a PIL Image with int values in 0 and 255."""
    np_array = torch_tensor.cpu().numpy()
    return numpy2image(np_array)


def image2tensor(pil_image, device="cpu"):
    """Convert a PIL Image to a torch tensor with values in 0 and 1."""
    np_array = image2numpy(pil_image)
    return torch.from_numpy(np_array).float().to(device)


def flip_image_vertically(image):
    """Flip a PIL Image vertically."""
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def flip_image_horizontally(image):
    """Flip a PIL Image horizontally."""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def tensor_img_bilinear_interp(img, points_2d):
    """bilinear interpolation of image values at points_2d
    
    args: 
        img (torch.tensor or np.ndarray): (H, W, C)
        points_2d (torch.tensor or np.ndarray): (N, 2) with values in [0, H-1], [0, W-1]
    
    out:
        interp_vals (torch.tensor or np.ndarray): (N, C)
    """
    
    # Extract dimensions from the image tensor
    H, W, C = img.shape[:3]
    
    # Get the number of points to interpolate
    N = points_2d.shape[0]
    
    # Create an array to store the interpolated values for each point
    interp_vals = torch.zeros(N, C, dtype=img.dtype, device=img.device)
    
    # Extract x and y coordinates from the points_2d array
    y, x = points_2d[:, 0], points_2d[:, 1]
    
    # Calculate the integer part of the coordinates for indexing
    y0, x0 = torch.floor(y).int(), np.floor(x).astype(int)
    
    # Calculate the neighboring pixel indices for interpolation
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    
    