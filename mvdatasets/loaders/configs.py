from mvdatasets.utils.printing import print_error


def get_scene_preset(dataset_name: str, scene_name: str) -> dict:

    # test dtu
    if dataset_name == "dtu":
        # dataset specific config
        config = {
            "subsample_factor": 1,
        }

    # test blended-mvs
    elif dataset_name == "blended-mvs":
        # dataset specific config
        config = {}

    # test blender
    elif dataset_name == "blender":
        # dataset specific config
        config = {
            "test_skip": 20,
        }

    # test shelly
    elif dataset_name == "shelly":
        # dataset specific config
        config = {"test_skip": 4, "init_sphere_radius_mult": 0.2}

    # test blendernerf
    elif dataset_name == "blendernerf":
        # dataset specific config
        config = {
            "test_skip": 10,
        }

    # test dmsr
    elif dataset_name == "dmsr":
        # dataset specific config
        config = {
            "test_skip": 5,
        }

    # test refnerf
    elif dataset_name == "refnerf":
        # dataset specific config
        config = {
            "test_skip": 10,
        }

    # test ingp
    elif dataset_name == "ingp":
        # dataset specific config
        config = {}

    # test llff
    elif dataset_name == "llff":
        # dataset specific config
        config = {
            "scene_type": "forward-facing",
        }

    # test mipnerf360
    elif dataset_name == "mipnerf360":

        # dataset specific config
        config = {
            "scene_type": "unbounded",
            "subsample_factor": 8,
        }

        # scene specific config
        if scene_name == "bicycle":
            config["rotate_scene_x_axis_deg"] = -104
            config["translate_scene_z"] = 0.1

        if scene_name == "garden":
            config["rotate_scene_x_axis_deg"] = -120
            config["translate_scene_z"] = 0.2

        if scene_name == "bonsai":
            config["rotate_scene_x_axis_deg"] = -130
            config["translate_scene_z"] = 0.25

        if scene_name == "counter":
            config["rotate_scene_x_axis_deg"] = -125
            config["translate_scene_y"] = -0.1
            config["translate_scene_z"] = 0.25

        if scene_name == "kitchen":
            config["rotate_scene_x_axis_deg"] = -130
            config["translate_scene_z"] = 0.2

        if scene_name == "room":
            config["rotate_scene_x_axis_deg"] = -115

        if scene_name == "stump":
            config["rotate_scene_x_axis_deg"] = -137
            config["translate_scene_z"] = 0.25

    # test d-nerf
    elif dataset_name == "d-nerf":
        # dataset specific config
        config = {}

    # test visor
    elif dataset_name == "visor":
        # dataset specific config
        config = {}

    # test panoptic-sports
    elif dataset_name == "panoptic-sports":
        # dataset specific config
        config = {}

    else:
        # undefined empty config
        config = {}

    return config
