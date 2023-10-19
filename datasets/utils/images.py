from PIL import Image
import numpy as np


def tensor2image(torch_tensor):
    """Convert a torch tensor with values between 0 and 1 and shape (H, W, C) to a PIL Image."""

    np_array = torch_tensor.cpu().numpy()
    return numpy2image(np_array)


def numpy2image(np_array):
    """Convert a numpy array with values between 0 and 1 and shape (H, W, C) to a PIL Image."""

    # if grayscale, repeat 3 times
    if np_array.shape[-1] == 1:
        # concat 3 times
        np_array = np.concatenate([np_array, np_array, np_array], axis=-1)

    return Image.fromarray(np.uint8(np_array * 255))


def image2numpy(pil_image):
    """Convert a PIL Image to a numpy array with values between 0 and 1 and shape (H, W, 3)."""
    return np.array(pil_image) / 255
