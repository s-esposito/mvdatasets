def is_dataset_supported(dataset_name):
    datasets_supported = ["dtu", "blended_mvs", "nerf_synthetic", "pac_nerf"]
    dataset_name = dataset_name.lower()
    if dataset_name in datasets_supported:
        return True
