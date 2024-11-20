# FROM: https://github.com/epic-kitchens/VISOR-VIS/blob/main/vis.py#L152

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import glob
import csv
import pandas as pd
import os
from tqdm import tqdm
import json
import collections, functools, operator
from PIL import Image
from scipy.stats import norm
from numpy import asarray


def do_stats_stage2_jsons_single_file_new(file):

    total_number_of_images = 0
    total_number_of_objects = 0
    # total_number_of_seq = 0
    total_number_objects_per_image = []
    objects = []
    infile = file
    f = open(infile)
    # returns JSON object as a dictionary
    data = json.load(f)

    # sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data["video_annotations"], key=lambda k: k["image"]["image_path"])

    total_number_of_images = total_number_of_images + len(data)

    # Iterating through the json list
    # index = 0
    # full_path=""
    # prev_seq = ""
    obj_per_image = 0
    for datapoint in data:
        obj_per_image = 0  # count number of objects per image
        # image_name = datapoint["image"]["name"]
        # image_path = datapoint["image"]["image_path"]
        masks_info = datapoint["annotations"]
        # generate_masks(image_name, image_path, masks_info, full_path) #this is for saving the same name (delete the if statemnt as well)
        entities = masks_info
        for entity in entities:  # loop over each object
            object_annotations = entity["segments"]
            if (
                not len(object_annotations) == 0
            ):  # if there is annotation for this object, add it
                total_number_of_objects = total_number_of_objects + 1
                objects.append(entity["name"])
                obj_per_image = obj_per_image + 1
        total_number_objects_per_image.append(obj_per_image)

    # print(objects)
    objects_counts = collections.Counter(objects)

    df = pd.DataFrame.from_dict(objects_counts, orient="index").reset_index()
    # print(df)
    # print("Number of sequences: ", (total_number_of_seq))
    # print("Number of images (masks): ", (total_number_of_images))
    # print("Number of unique objects: ", len(set(objects)))

    return objects_counts.most_common()


def imwrite_indexed_2(filename, im, non_empty_objects=None):

    davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(
        np.uint8
    )
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
    color_palette = davis_palette
    assert len(im.shape) < 4 or im.shape[0] == 1  # requires batch size 1
    im = torch.from_numpy(im)
    im = Image.fromarray(im.detach().cpu().squeeze().numpy(), "P")
    im.putpalette(color_palette.ravel())
    im.save(filename)


def overlay_semantic_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):

    d = colors
    colors = colors.ravel()
    a = ann
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError("First two dimensions of `im` and `ann` must match")
    if im.shape[-1] != 3:
        raise ValueError("im must have three channels at the 3 dimension")

    # ann2 = a.convert('RGB')
    ann2 = np.array(a)

    colors = np.asarray(colors, dtype=np.uint8)

    # mask = colors[ann2]
    mask = d[ann2]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours(
                (ann == obj_id).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )[-2:]
            cv2.drawContours(
                img, contours[0], -1, d[obj_id].tolist(), contour_thickness
            )
    return img


def generate_masks_for_image(
    image_name,
    image_path,
    masks_info,
    output_directory,
    object_keys=None,
    is_overlay=False,
    images_root_directory=".",
    input_resolution=(1920, 1080),
    output_resolution=(1920, 1080),
):

    non_empty_objects = []

    img = np.zeros([input_resolution[1], input_resolution[0]], dtype=np.uint8)

    # masks_info = sorted(masks_info, key=lambda k: k['name'])
    # index = 1
    # if (image_path == 'P30_107/Part_008/P30_107_seq_00072/frame_0000038529/frame_0000038529.jpg'): #'P30_107/Part_008/P30_107_seq_00067/frame_0000037246/frame_0000037246.jpg'

    entities = masks_info
    i = 1
    for entity in entities:
        object_annotations = entity["segments"]
        polygons = []
        polygons.append(object_annotations)
        non_empty_objects.append(entity["name"])
        ps = []
        for polygon in polygons:
            for poly in polygon:
                if poly == []:
                    poly = [[0.0, 0.0]]
                ps.append(np.array(poly, dtype=np.int32))
        if object_keys:
            if entity["name"] in object_keys.keys():
                # print(ps)
                # print(entity['name'])
                # print(object_keys[entity['name']])
                cv2.fillPoly(
                    img,
                    ps,
                    (
                        object_keys[entity["name"]],
                        object_keys[entity["name"]],
                        object_keys[entity["name"]],
                    ),
                )
                # cv2.polylines(img, ps, True, (255,255,255), thickness=1)
        else:
            cv2.fillPoly(img, ps, (i, i, i))
        i += 1
        # visualize(img)
        # if (not np.all(img == 0)):  # image_path.__contains__("P03_120_seq_00064")

    image_name = image_name.replace("jpg", "png")
    if not np.all(img == 0):
        image_name = image_name.replace("jpg", "png")
        # print(output_directory + image_name)
        # cv2.imwrite(output_directory+image_name,img)
        image_data = asarray(img)
        if input_resolution != output_resolution:
            out_image = cv2.resize(
                image_data,
                (output_resolution[0], output_resolution[1]),
                interpolation=cv2.INTER_NEAREST,
            )
            out_image = (np.array(out_image)).astype("uint8")
        else:
            out_image = image_data

        imwrite_indexed_2(os.path.join(output_directory, image_name), out_image)

        if is_overlay:
            image1 = image_name.replace("png", "jpg")
            video = "_".join(image_name.split("_")[:2])
            image1_overlay = Image.open(
                os.path.join(images_root_directory, video + "/" + image1)
            )

            if image1_overlay.size != output_resolution:
                image1_overlay = image1_overlay.resize(output_resolution)

            davis_palette = np.repeat(
                np.expand_dims(np.arange(0, 256), 1), 3, 1
            ).astype(np.uint8)
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

            a = overlay_semantic_mask(
                image1_overlay,
                out_image,
                colors=davis_palette,
                alpha=0.2,
                contour_thickness=1,
            )
            img2 = Image.fromarray(a, "RGB")
            img2.save(os.path.join(output_directory, image_name))

    # imwrite_indexed(output_directory + image_name, img,non_empty_objects)
