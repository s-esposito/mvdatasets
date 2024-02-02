def is_dataset_supported(dataset_name):
    # TODO: add more datasets
    # # llff, tanks_and_temples, ...
    # pac_nerf currently not supported as major changes are needed
    datasets_supported = [
                            "dtu",
                            "blender",
                            "blendernerf",
                            "dmsr",
                            "refnerf",
                            "llff",
                            "mipnerf360"
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
        pc_paths = ["debug/meshes/dtu/dtu_scan83.ply"]
        config = {}

    # test blender
    if dataset_name == "blender":
        scene_name = "lego"
        pc_paths = ["debug/point_clouds/blender/lego.ply"]
        config = {}

    # test blendernerf
    if dataset_name == "blendernerf":
        scene_name = "plushy"
        pc_paths = ["debug/meshes/blendernerf/plushy.ply"]
        config = {
            "load_mask": 1,
            "scene_scale_mult": 0.4,
            "white_bg": 1,
            "test_skip": 10,
            "subsample_factor": 1.0
        }

    # test dmsr
    if dataset_name == "dmsr":
        scene_name = "dinning"
        pc_paths = ["debug/meshes/dmsr/dinning.ply"]
        config = {
            "test_skip": 10,
            "scene_scale_mult": 0.4
        }

    # test refnerf
    if dataset_name == "refnerf":
        scene_name = "car"
        pc_paths = []
        config = {
            "test_skip": 10,
        }
        
    # test llff
    if dataset_name == "llff":
        scene_name = "fern"
        pc_paths = ["debug/point_clouds/llff/fern.ply"]
        config = {
            "subsample_factor": 1.0,
            "scene_scale_mult": 0.03,
            "rotate_scene_x_axis_deg": 90.0
        }
    
    # test mipnerf360
    if dataset_name == "mipnerf360":
        scene_name = "bicycle"
        pc_paths = ["debug/point_clouds/mipnerf360/bicycle.ply"]
        config = {
            "subsample_factor": 4.0,
            "scene_scale_mult": 0.03,
        }
    
    return scene_name, pc_paths, config