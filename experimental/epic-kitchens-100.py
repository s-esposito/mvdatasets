import csv
import glob
import os
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path

datasets_path = "/home/stefano/Data/"
dataset_name = "epic-kitchens-100"
sequence_name = "P01"
dataset_path = Path(datasets_path) / dataset_name
output_directory = "experimental/output"

jsons_path = dataset_path / "JSON_DATA"
jsons_files_paths = sorted(glob.glob(os.path.join(jsons_path, "*.json")))
# print(jsons_files_paths)

# load json file for the first sequence
json_file_path = jsons_files_paths[0]
# load json file
f = open(json_file_path)
# returns JSON object as a dictionary
data = json.load(f)
# sort based on the folder name (to guarantee to start from its first frame of each sequence)
print(data.keys())
camera = data["camera"]
print(camera["id"])
model = camera["model"]  # OPENCV
width = camera["width"]
height = camera["height"]
params = camera["params"]
print(params)
# data["images"] are images_files_names
points = np.array(data["points"])[:, :3]
# what's np.array(data["points"])[:, 3] ?
# what's np.array(data["points"])[:, 4] ?
print(points.shape)