def load_llff(
    data_path,
    load_mask=True,
    rotate_scene_x_axis_deg=0,
    scene_scale_mult=0.25,  # keeps the object within the unit sphere
    subsample_factor=1,
    # no_ndc = False, # do not use normalized device coordinates (set for non-forward facing scenes)
    # args.factor, recenter=True, bd_factor=.75, spherify=args.spherify
):
    near = 0.0
    far = 1.0

    pass
