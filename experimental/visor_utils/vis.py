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
from mvdatasets.visualization.colormaps import davis_palette


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

    # image_name = image_name.replace("jpg", "png")
    # if not np.all(img == 0):
    #     image_name = image_name.replace("jpg", "png")
    #     # print(output_directory + image_name)
    #     # cv2.imwrite(output_directory+image_name,img)
    #     image_data = asarray(img)
    #     # if input_resolution != output_resolution:
    #     #     out_image = cv2.resize(
    #     #         image_data,
    #     #         (output_resolution[0], output_resolution[1]),
    #     #         interpolation=cv2.INTER_NEAREST,
    #     #     )
    #     #     out_image = (np.array(out_image)).astype("uint8")
    #     # else:
    #     out_image = image_data

    # imwrite_indexed_2(os.path.join(output_directory, image_name), out_image)

    # if is_overlay:
    #     image1 = image_name.replace("png", "jpg")
    #     video = "_".join(image_name.split("_")[:2])
    #     image1_overlay = Image.open(
    #         os.path.join(images_root_directory, video + "/" + image1)
    #     )

    #     if image1_overlay.size != output_resolution:
    #         image1_overlay = image1_overlay.resize(output_resolution)

    #     a = overlay_semantic_mask(
    #         image1_overlay,
    #         out_image,
    #         colors=davis_palette,
    #         alpha=0.2,
    #         contour_thickness=1,
    #     )
    #     img2 = Image.fromarray(a, "RGB")
    #     img2.save(os.path.join(output_directory, image_name))

    # imwrite_indexed(output_directory + image_name, img,non_empty_objects)
