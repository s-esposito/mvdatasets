import numpy as np

# from https://github.com/epic-kitchens/VISOR-VIS/blob/main/vis.py
davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
davis_palette[:104, :] = [
    [0, 0, 0],
    [200, 0, 0],
    [0, 200, 0],
    [200, 128, 0],
    [0, 0, 200],
    [200, 0, 200],
    [0, 200, 200],
    [200, 200, 200],
    [252, 93, 82],
    [160, 121, 99],
    [164, 188, 119],
    [0, 60, 29],
    [75, 237, 255],
    [148, 169, 183],
    [96, 74, 207],
    [255, 186, 255],
    [255, 218, 231],
    [136, 30, 23],
    [231, 181, 131],
    [219, 226, 216],
    [0, 196, 107],
    [0, 107, 119],
    [0, 125, 227],
    [153, 134, 227],
    [91, 0, 56],
    [86, 0, 7],
    [246, 207, 195],
    [87, 51, 0],
    [125, 131, 122],
    [187, 237, 218],
    [46, 57, 59],
    [164, 191, 255],
    [37, 29, 57],
    [144, 53, 104],
    [79, 53, 54],
    [255, 163, 128],
    [255, 233, 180],
    [68, 100, 62],
    [0, 231, 199],
    [0, 170, 233],
    [0, 20, 103],
    [195, 181, 219],
    [148, 122, 135],
    [200, 128, 129],
    [46, 20, 10],
    [86, 78, 24],
    [180, 255, 188],
    [0, 36, 33],
    [0, 101, 139],
    [50, 60, 111],
    [188, 81, 205],
    [168, 9, 70],
    [167, 91, 59],
    [35, 32, 0],
    [0, 124, 28],
    [0, 156, 145],
    [0, 36, 57],
    [0, 0, 152],
    [89, 12, 97],
    [249, 145, 183],
    [255, 153, 170],
    [255, 153, 229],
    [184, 143, 204],
    [208, 204, 255],
    [11, 0, 128],
    [69, 149, 230],
    [82, 204, 194],
    [77, 255, 136],
    [6, 26, 0],
    [92, 102, 41],
    [102, 85, 61],
    [76, 45, 0],
    [229, 69, 69],
    [127, 38, 53],
    [128, 51, 108],
    [41, 20, 51],
    [25, 16, 3],
    [102, 71, 71],
    [77, 54, 71],
    [143, 122, 153],
    [42, 41, 51],
    [4, 0, 51],
    [31, 54, 77],
    [204, 255, 251],
    [51, 128, 77],
    [61, 153, 31],
    [194, 204, 143],
    [255, 234, 204],
    [204, 119, 0],
    [204, 102, 102],
    [64, 0, 0],
    [191, 0, 0],
    [64, 128, 0],
    [191, 128, 0],
    [64, 0, 128],
    [191, 0, 128],
    [64, 128, 128],
    [191, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 191, 0],
    [128, 191, 0],
    [0, 64, 128],
    [128, 64, 128],
]  # first 90 for the regular colors and the last 14 for objects having more than one segment
