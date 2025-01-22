import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from mvdatasets import MVDataset
from mvdatasets import Camera
from mvdatasets.utils.printing import print_warning, print_info, print_error


# evaluates the model
@torch.no_grad()
def eval_rendered_imgs(renders_path, scene_name):
    """

    Returns:
        list of results (PerSceneEvaluator), one for each render mode
    """
    # iterate over folders in renders_path
    # each folder contains a different render mode
    # (e.g. "volumetric", "sphere_traced", ...)

    # check if path exists
    if not os.path.exists(renders_path):
        raise ValueError(f"renders path {renders_path} does not exist")

    # # list all folders in renders_path
    # render_modes = []
    # for name in os.listdir(renders_path):
    #     if os.path.isdir(os.path.join(renders_path, name)):
    #         render_modes.append(name)
    # print_info(f"found renders for rendering modalities: {render_modes}")

    render_modes_paths = [os.path.join(renders_path, folder) for folder in render_modes]

    results = []

    # unmasked
    for render_mode_path, render_mode in zip(render_modes_paths, render_modes):
        # print(f"evaluating render mode {render_mode}")

        # check if "gt" and "rgb" folders exists
        if os.path.exists(os.path.join(render_mode_path, "gt")) and os.path.exists(
            os.path.join(render_mode_path, "rgb")
        ):
            #

            # get all images filenames in gt
            # "000.png", "001.png", ... "999.png"
            img_filenames = os.listdir(os.path.join(render_mode_path, "gt"))
            # sort by name
            img_filenames.sort()

            # list all images in gt
            gt_path = os.path.join(render_mode_path, "gt")
            gt_imgs_paths = sorted(
                [os.path.join(gt_path, img_filename) for img_filename in img_filenames]
            )

            # load corresponding images in "rgb"
            rgb_path = os.path.join(render_mode_path, "rgb")
            pred_imgs_paths = sorted(
                [os.path.join(rgb_path, img_filename) for img_filename in img_filenames]
            )

            # load images and compute psnr, ssim, lpips

            scene_evaluator = PerSceneEvaluator(render_mode)
            scene_masked_evaluator = PerSceneEvaluator(render_mode + "_masked")

            print(f"[bold black]evaluating {render_mode}[/bold black]")
            print("[bold black]img_name, psnr, ssim, lpips[/bold black]")
            for img_filename, gt_img_path, pred_img_path in zip(
                img_filenames, gt_imgs_paths, pred_imgs_paths
            ):

                img_name = img_filename.split(".")[0]

                gt_img_pil = Image.open(gt_img_path)
                pred_img_pil = Image.open(pred_img_path)

                gt_rgb = image_to_tensor(gt_img_pil).cuda()
                pred_rgb = image_to_tensor(pred_img_pil).cuda()

                gt_rgb_tensor = gt_rgb.cuda()
                pred_rgb_tensor = pred_rgb.cuda()
                gt_rgb_tensor = gt_rgb_tensor.permute(2, 0, 1).unsqueeze(0)
                pred_rgb_tensor = pred_rgb_tensor.permute(2, 0, 1).unsqueeze(0)

                psnr_val = psnr(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
                ssim_val = ssim(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
                lpips_val = LPIPS()(pred_rgb_tensor, gt_rgb_tensor).item()

                print(
                    f"[bold black]{img_name}[/bold black]",
                    psnr_val,
                    ssim_val,
                    lpips_val,
                )

                scene_evaluator.update(img_name, psnr_val, ssim_val, lpips_val)

            results.append(scene_evaluator)

    return results


# folders are same as 3DGS
# RUN-ROOT
# ├── datasets


def visual_evaluation(
    mv_data: MVDataset,
    run_path: Path,
    save_path: Optional[Path] = None,  # if set, saves the results to the given path
) -> None:
    """
    todo
    """

    cameras_splits = mv_data.get_splits()

    # prepare output dict
    eval_res = {}
    for split in cameras_splits:
        eval_res[split] = dict()

    # run evaluation for each split
    for split, eval_dict in eval_res.items():
        print(f"\nrunning rendering on {split} set")

        # check if save_path/split.csv exists
        if os.path.exists(os.path.join(save_path, f"{split}.csv")):

            print_warning(f"{split} results exists in {run_path}, skipping evaluation")

        else:

            print(
                f"[bold blue]INFO[/bold blue]: evaluating {split} renders in {run_path}"
            )

            if os.path.exists(os.path.join(run_path, split)):
                print_info(f"found renders for {split}")
            else:
                raise ValueError(f"renders for {split} not found in {run_path}")

            render_modes_eval_res = eval_rendered_imgs(
                os.path.join(run_path, split), scene_name=mv_data.scene_name
            )

            for res in render_modes_eval_res:
                res_avg = res.results_averaged()
                eval_dict.update(res_avg)
                # print results
                print(f"render mode: {res.render_mode}")
                for key, value in res_avg.items():
                    print(f"{key}: {value}")
                # store results to csv
                res.save_to_csv(os.path.join(run_path, split))

            # create results dir if not exists
            os.makedirs(save_path, exist_ok=True)

            # save results to csv
            # TODO: assumption: only one render mode is evaluated
            for res in render_modes_eval_res:
                res_avg = res.results_averaged()
                eval_dict.update(res_avg)
                # print results
                print(f"render mode: {res.render_mode}")
                for key, value in res_avg.items():
                    print(f"{key}: {value}")
                # store results to csv
                res.save_to_csv(save_path, override_filename=f"{split}")

    return eval_res
