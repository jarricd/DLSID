import json
import pyiqa
import argparse
import logging
import pathlib
import sys
import torch
import numpy as np
import datetime
import cv2

logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler("metrics.log"),
    logging.StreamHandler()
])

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Calculate metrics for LLLR and NLHR images.",
                                              epilog="This program does not have supercow powers.")
    argument_parser.add_argument("input_path_NLHR", help="Input NLHR images for calculations.")
    argument_parser.add_argument("input_path_LLLR", help="Input LLLR images for calculations.")

    args = argument_parser.parse_args()

    input_path_NLHR = pathlib.Path(args.input_path_NLHR)
    input_path_LLLR = pathlib.Path(args.input_path_LLLR)

    if not input_path_NLHR.is_dir():
        sys.exit(1)

    if not input_path_LLLR.is_dir():
        sys.exit(2)

    # img_files_NLHR = []
    # img_files_NLHR.extend(list(input_path_NLHR.glob("*jpg")))
    # # img_files_NLHR.extend(list(input_path_NLHR.glob("*png")))
    # img_files_NLHR.sort()

    img_files_LLLR = []
    # img_files_LLLR.extend(list(input_path_LLLR.glob("*jpg")))
    img_files_LLLR.extend(list(input_path_LLLR.glob("*png")))
    # img_files_LLLR.sort()
    metrics = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    psnr_metric = pyiqa.create_metric("psnr").to(device)
    ssim_metric = pyiqa.create_metric("ssim").to(device)
    lpip_metric = pyiqa.create_metric("lpips").to(device)
    # print(img_files_NLHR)
    # print(img_files_LLLR)

    for img_lllr in img_files_LLLR: 
        img_nlhr = pathlib.Path(f"{input_path_NLHR}/{img_lllr.stem}{img_lllr.suffix}")
        logging.info(f"Calculating scores for {img_lllr} and {img_nlhr}")
        try:
            psnr_score = np.round(psnr_metric(str(img_nlhr), str(img_lllr)).item(), 4)
            ssim_score = np.round(ssim_metric(str(img_nlhr), str(img_lllr)).item(), 4)
            lpip_score = np.round(lpip_metric(str(img_nlhr), str(img_lllr)).item(), 4)
        except AssertionError:
            continue
        metrics.update({f"{img_nlhr.stem}{img_nlhr.suffix}": {"ssim": ssim_score, "psnr": psnr_score, "lpips": lpip_score}})

    output_json = json.dumps(metrics)
    with open(f"scores_iqa_{datetime.datetime.now()}.json", "w") as f:
        f.write(output_json)