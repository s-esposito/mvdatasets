from mvdatasets.utils.printing import print_error


datasets_path = "/home/stefano/Data"


def is_dataset_supported(dataset_name):
    datasets_supported = [
                            "dtu",
                            "blender",
                            "ingp",
                            "blendernerf",
                            "dmsr",
                            "refnerf",
                            "llff",
                            "mipnerf360",
                            "shelly"
                        ]
    dataset_name = dataset_name.lower()
    if dataset_name in datasets_supported:
        return True
    else:
        return False


def get_dataset_test_preset(dataset_name):

    if not is_dataset_supported(dataset_name):
        print_error(f"{dataset_name} is not a supported dataset.")
    
    # test DTU
    if dataset_name == "dtu":
        scene_name = "dtu_scan83"
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]
        # dataset specific config
        config = {}

    # test blender
    if dataset_name == "blender":
        scene_name = "lego"
        pc_paths = ["tests/assets/point_clouds/blender/lego.ply"]
        # dataset specific config
        config = {
            "test_skip": 20,
        }
        
    # test shelly
    if dataset_name == "shelly":
        scene_name = "khady"
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]
        # dataset specific config
        config = {
            "test_skip": 4,
            "init_sphere_scale": 0.2
        }

    # test blendernerf
    if dataset_name == "blendernerf":
        scene_name = "plushy"
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]
        # dataset specific config
        config = {
            "test_skip": 10,
        }

    # test dmsr
    if dataset_name == "dmsr":
        scene_name = "dinning"
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]
        # dataset specific config
        config = {
            "test_skip": 5,
        }

    # test refnerf
    if dataset_name == "refnerf":
        scene_name = "car"
        pc_paths = []
        # dataset specific config
        config = {
            "test_skip": 10,
        }
        
    # test ingp
    if dataset_name == "ingp":
        scene_name = "fox"
        pc_paths = []
        # dataset specific config
        config = {}
        
    # test llff
    if dataset_name == "llff":
        scene_name = "fern"
        pc_paths = ["tests/assets/point_clouds/llff/fern.ply"]
        # dataset specific config
        config = {
            "scene_type": "forward_facing",
        }
    
    # test mipnerf360
    if dataset_name == "mipnerf360":
        scene_name = "garden"
        pc_paths = []
        
        # dataset specific config
        config = {
            "scene_type": "unbounded",
            "subsample_factor": 4,
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
    
    return scene_name, pc_paths, config