import os
import json


def load_blender(
    data_path,
    load_mask=True,
    rotate_scene_x_axis_deg=0,
    scene_scale_mult=0.25,  # keeps the object within the unit sphere
    subsample_factor=1,
    white_bg=True,  # remove alpha channel and replace with white background
    test_skip=1,
):
    # cameras objects
    cameras = []

    # only load images from the train split
    s = "train"
    with open(os.path.join(data_path, "transforms_{}.json".format(s)), "r") as fp:
        metas = json.load(fp)

    if white_bg:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    else:
        images = images[..., :3]

    pass
