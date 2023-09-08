import json
import torch
import numpy as np
import os
from tqdm import tqdm
import imageio
import cv2 as cv

def load_data(data_path, H=800, W=800):

    # TODO Stefano: read automatically
    # H, W also
    num_views = 11
        
    with open(os.path.join(data_path, "all_data.json")) as f:
        data_info = json.load(f)

    n_frames = int(len(data_info) / num_views) - 1

    c2w_all = torch.zeros(num_views, 3, 4)
    K_all = torch.zeros(num_views, 3, 3)
    rgb_all = torch.zeros(num_views, n_frames, H, W, 3)

    for entry in tqdm(data_info):
        
        cam_id, frame_id = [int(i) for i in entry["file_path"].split("/")[-1].rstrip(".png").lstrip("r_").split("_")]
        
        if frame_id < 0:
            continue
        
        c2w_all[cam_id] = torch.tensor(entry["c2w"])
        K_all[cam_id] = torch.tensor(entry["intrinsic"])
        img = np.array(imageio.imread(os.path.join(data_path, entry["file_path"])))[..., :3]

        cv.imwrite(os.path.join(data_path, f"data/m_{cam_id}_{frame_id}.png"), img[...,::-1])
        rgb_all[cam_id, frame_id] = torch.from_numpy(img.astype(np.float32) / 255.)
        
    return c2w_all, K_all, rgb_all