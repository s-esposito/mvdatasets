def is_dataset_supported(dataset_name):
    datasets_supported = [
                            "dtu",
                            "blender",
                            "ingp",
                            "blendernerf",
                            "dmsr",
                            "refnerf",
                            # "llff",
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
        raise NotImplementedError(f"Dataset {dataset_name} not currently supported.")
    
    # test DTU
    if dataset_name == "dtu":
        scene_name = "dtu_scan83"
        pc_paths = [f"debug/meshes/{dataset_name}/{scene_name}.ply"]
        config = {}

    # test blender
    if dataset_name == "blender":
        scene_name = "lego"
        pc_paths = ["debug/point_clouds/blender/lego.ply"]
        config = {
            "test_skip": 20
        }
        
    # test shelly
    if dataset_name == "shelly":
        scene_name = "khady"
        pc_paths = [f"debug/meshes/{dataset_name}/{scene_name}.ply"]
        config = {
            "test_skip": 4,
            "scene_radius": 1.2
        }

    # test blendernerf
    if dataset_name == "blendernerf":
        scene_name = "plushy"
        pc_paths = [f"debug/meshes/{dataset_name}/{scene_name}.ply"]
        config = {
            "test_skip": 10
        }

    # test dmsr
    if dataset_name == "dmsr":
        scene_name = "dinning"
        pc_paths = [f"debug/meshes/{dataset_name}/{scene_name}.ply"]
        config = {
            "test_skip": 5
        }

    # test refnerf
    if dataset_name == "refnerf":
        scene_name = "car"
        pc_paths = []
        config = {
            "test_skip": 10,
        }
        
    # test ingp
    if dataset_name == "ingp":
        scene_name = "fox"
        pc_paths = []
        config = {}
        
    # test llff
    if dataset_name == "llff":
        scene_name = "fern"
        pc_paths = ["debug/point_clouds/llff/fern.ply"]
        config = {
            "scene_type": "forward_facing",
            "scene_scale_mult": 0.03,
        }
    
    # test mipnerf360
    if dataset_name == "mipnerf360":
        scene_name = "bicycle"
        pc_paths = []
        
        # dataset specific config
        config = {
            "scene_type": "unbounded",
            "subsample_factor": 8,
            "scene_scale_mult": 0.1,
        }
        
        # scene specific config
        if scene_name == "bicycle":
            config["rotate_scene_x_axis_deg"] = -104
        if scene_name == "garden":
            config["rotate_scene_x_axis_deg"] = -120
    
    return scene_name, pc_paths, config