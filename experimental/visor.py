import csv
import glob
import os
import json
from tqdm import tqdm
from pathlib import Path
from visor_utils.vis import (
    do_stats_stage2_jsons_single_file_new,
    generate_masks_for_image
)

datasets_path = "/home/stefano/Data/"
dataset_name = "visor"
sequence_name = "P01"
dataset_path = Path(datasets_path) / dataset_name
output_directory = "experimental/output"

# read classes
classes_file_name = "EPIC_100_noun_classes_v2.csv"
classes_file_path = dataset_path / classes_file_name
with open(classes_file_path, mode="r") as file:
    reader = csv.reader(file)
    rows = list(reader)
id_keys = {}  # maps ids to keys
id_instances = {}  # maps ids to instances
category_ids = {}  # maps categories to lists of ids
for row in rows[1:]:
    id = row[0]
    key = row[1]
    instances = row[2]
    category = row[3]
    id_keys[id] = key
    id_instances[id] = instances
    if category not in category_ids:
        category_ids[category] = []
    category_ids[category].append(id)
# print(id_keys)
# print(category_ids)

# dataset loading

jsons_path = dataset_path / "GroundTruth-SparseAnnotations" / "annotations" / "train"
json_files_paths = sorted(glob.glob(os.path.join(jsons_path, "*.json")))

json_file_path = json_files_paths[0]
print(json_file_path)

object_keys = {}
objects = do_stats_stage2_jsons_single_file_new(json_file_path)
# print('objects: ',objects)
i = 1
for key, _ in objects:
    object_keys[key] = i
    i = i + 1
max_count = max(object_keys.values())

# read json file
f = open(json_file_path)
# returns JSON object as a dictionary
data = json.load(f)
# sort based on the folder name (to guarantee to start from its first frame of each sequence)
data = sorted(data["video_annotations"], key=lambda k: k["image"]["image_path"])

for datapoint in data:
    image_name = datapoint["image"]["name"]
    image_path = datapoint["image"]["image_path"]
    masks_info = datapoint["annotations"]
    full_path = os.path.join(
        output_directory, image_path.split("/")[0] + "/"
    )  # until the end of sequence name
    print(image_name)
    print(image_path)

    generate_masks_for_image(
        image_name,
        image_path,
        masks_info,
        full_path,
        object_keys=object_keys,
        is_overlay=False,
        images_root_directory='.',
        input_resolution=(1920, 1080),
        output_resolution=(1920, 1080),
    )  # this is for unique id for each object throughout the video

    exit(0)

# for json_file in tqdm(json_files_names):
#     input_resolution = (1920, 1080)
#     frame_rate = 3
