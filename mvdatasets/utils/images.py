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
