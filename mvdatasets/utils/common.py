def is_dataset_supported(dataset_name):
    # TODO: add more datasets
    # # llff, tanks_and_temples, ...
    # pac_nerf currently not supported as major changes are needed
    datasets_supported = [
                            "dtu",
                            "blender",
                            "blendernerf"
                        ]
    dataset_name = dataset_name.lower()
    if dataset_name in datasets_supported:
        return True
