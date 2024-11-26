import csv
import glob
import os
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from mvdatasets.loaders.dynamic.visor import (
    _generate_mask_from_polygons,
    _generate_semantic_mask_from_polygons
)
from mvdatasets.visualization.colormaps import davis_palette
import matplotlib.pyplot as plt

datasets_path = "/home/stefano/Data/"
dataset_name = "visor"
sequence_name = "P01"
dataset_path = Path(datasets_path) / dataset_name
output_directory = "experimental/output"

# dataset loading

rgbs_path = dataset_path / "GroundTruth-SparseAnnotations" / "rgb_frames" / "train"
jsons_path = dataset_path / "GroundTruth-SparseAnnotations" / "annotations" / "train"
json_files_paths = sorted(glob.glob(os.path.join(jsons_path, "*.json")))

# get the first json file
json_file_path = json_files_paths[0]
print(json_file_path)

# read json file
f = open(json_file_path)
# returns JSON object as a dictionary
data = json.load(f)
# sort based on the folder name (to guarantee to start from its first frame of each sequence)
video_data = sorted(data["video_annotations"], key=lambda k: k["image"]["image_path"])

pbar = tqdm(video_data, desc="frames", ncols=100)
for frame_data in pbar:
    
    frame_data = video_data[100]
    
    video_name = frame_data["image"]["video"]
    image_name = frame_data["image"]["name"]
    # image_path = frame_data["image"]["image_path"]
    dir_name = image_name.split("_")[0]
    print("video_name", video_name)
    print("image_name", image_name)
    
    # real rgb path
    image_path = rgbs_path / dir_name / image_name
    img_pil = Image.open(image_path)
    img_np = np.array(img_pil)[..., :3]
    # plt.imshow(img_np)
    # plt.show()
    
    annotations_list = frame_data["annotations"]
    
    # # mask
    # mask = _generate_mask_from_polygons(
    #     annotations_list,
    #     width=1920,
    #     height=1080
    # )
    # plt.imshow(mask)
    # plt.show()
    
    # semantic mask
    semantic_mask = _generate_semantic_mask_from_polygons(
        annotations_list,
        width=1920,
        height=1080
    )
    # # get max value in the semantic mask
    # max_value = np.max(semantic_mask)
    # print("max_value", max_value)
    # colours = plt.cm.get_cmap('viridis', max_value + 1)
    # cmap = colours(np.linspace(0, 1, max_value + 1))  # Obtain RGB colour map
    # print(cmap.shape)
    # cmap[0, -1] = 0  # Set alpha for label 0 to be 0
    # cmap[1:, -1] = 1.0  # Set the other alphas for the labels to be 0.3
    
    colored_semantic_mask = davis_palette[semantic_mask.flatten()]
    # colored_semantic_mask = cmap[semantic_mask.flatten()]
    colored_semantic_mask = colored_semantic_mask.reshape(semantic_mask.shape[0], semantic_mask.shape[1], -1)
    print(colored_semantic_mask.shape)
    
    plt.imshow(colored_semantic_mask)
    plt.show()
    
    exit(0)

# # read classes
# classes_file_name = "EPIC_100_noun_classes_v2.csv"
# classes_file_path = dataset_path / classes_file_name
# with open(classes_file_path, mode="r") as file:
#     reader = csv.reader(file)
#     rows = list(reader)
# id_keys = {}  # maps ids to keys
# id_instances = {}  # maps ids to instances
# category_ids = {}  # maps categories to lists of ids
# for row in rows[1:]:
#     id = row[0]
#     key = row[1]
#     instances = row[2]
#     category = row[3]
#     id_keys[id] = key
#     id_instances[id] = instances
#     if category not in category_ids:
#         category_ids[category] = []
#     category_ids[category].append(id)
# # print(id_keys)
# # print(category_ids)



# # get all objects in the current sequence
# object_keys = {}
# objects = do_stats_stage2_jsons_single_file_new(json_file_path)
# print("objects:", objects)
# i = 1
# for key, _ in objects:
#     object_keys[key] = i
#     i = i + 1
# max_count = max(object_keys.values())
# # nr of objects in the current sequence
# print("max_count:", max_count)

