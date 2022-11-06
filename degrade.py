"""Perform series of degradations."""
import argparse
import logging
import cv2
import degrad.degradations
import random
import sys
import torch
import quality.calculate
import json
from pathlib import Path
from basicsr.archs.zerodce_arch import ConditionZeroDCE


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate LLLR images out of target dataset.",
                                         epilog="This program does not have supercow powers.")
    arg_parser.add_argument("dset_path", help="Path of input dataset.")
    arg_parser.add_argument("output_path", help="Path of output dataset.")
    arg_parser.add_argument("--noise_amp", help="Amplify noise by supplied factor.", default=1, type=float)
    arg_parser.add_argument("--degrad_count", help="How many degradations should be used?", default=3, type=int)
    arg_parser.add_argument("--zero_dce_pth", help="Zero-dce ckpt path", default="LEDNet/weights/ce_zerodce.pth")
    args = arg_parser.parse_args()

    dset_path_obj = Path(args.dset_path)
    output_dset_obj = Path(args.output_path)
    img_files = []
    if not dset_path_obj.is_dir():
        logging.error("Provided dset path is not a directory. Exiting.")
        sys.exit(1)

    if not output_dset_obj.exists():
        # create dir
        output_dset_obj.mkdir()

    if not output_dset_obj.is_dir():
        logging.error("Provided output path is not a directory. Exiting.")
        sys.exit(2)

    img_files.extend(list(dset_path_obj.glob("*jpg")))
    img_files.extend(list(dset_path_obj.glob("*png")))

    degrad_count = args.degrad_count
    blur_types = ["avg", "gauss", "median", "bilat"]

    # prep CE-ZeroDCE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.zero_dce_pth
    net = ConditionZeroDCE().to(device)
    if str(device) == "cpu":
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint)
    net.eval()

    scores = {}
    degradations_lut = {
        0: "exposure_decrease",
        1: "poisson_noise",
        2: "speckle_noise",
        3: "gauss_noise",
        4: "add_jpeg_noise",
        5: "add_webp_noise",
        6: "downsample",
        7: "resize",
        8: "blur"
    }

    for inter_num, target_img in enumerate(img_files):
        print(f"Degrading img {target_img}")
        exp_range = [0.1, 0.5]
        applied_degradations = {}
        target_exposure = random.uniform(*exp_range)
        degraded_img = cv2.imread(str(target_img))
        original_img = degraded_img.copy()
        degraded_img = degrad.degradations.zero_dce_exposure(net, degraded_img, target_exposure, device)
        applied_degradations.update({degradations_lut[0]: {"exposure_decrease": target_exposure, "threshold": 0.97}})
        for iterations in range(0, degrad_count):
            selected_degrad = random.randint(0, 7)
            if selected_degrad == 0:
                degraded_img, poisson_seed = degrad.degradations.add_poisson_noise(degraded_img, args.noise_amp)
                applied_degradations.update({"poisson_noise": {"noise_amp": args.noise_amp, "seed": poisson_seed}})
            elif selected_degrad == 1:
                low_lvl = random.randint(2, 5)
                upper_lvl = random.randint(10, 25)
                degraded_img, rnum, seed = degrad.degradations.add_speckle_noise(degraded_img, low_lvl, upper_lvl)
                applied_degradations.update({"speckle_noise": {"low_lvl": low_lvl, "upper_lvl": upper_lvl, "rnum": rnum, "seed": seed}})
            elif selected_degrad == 2:
                low_lvl = random.randint(2, 5)
                upper_lvl = random.randint(10, 25)
                degraded_img, rnum, seed = degrad.degradations.add_gauss_noise(degraded_img, low_lvl, upper_lvl)
                applied_degradations.update({"gauss_noise": {"low_lvl": low_lvl, "upper_lvl": upper_lvl, "rnum": rnum, "seed": seed}})
            elif selected_degrad == 3 and target_img.suffix != ".jpg":  # no point adding more jpeg noise to a jpeg img
                degraded_img, quality_comp = degrad.degradations.add_jpeg_noise(degraded_img)
                applied_degradations.update({"jpeg_noise": {"quality": quality_comp}})
            elif selected_degrad == 4:
                degraded_img, quality_comp = degrad.degradations.add_webp_noise(degraded_img)
                applied_degradations.update({"web_noise": {"quality": quality_comp}})
            elif selected_degrad == 5:
                degraded_img = degrad.degradations.downsample(degraded_img, 2, retain_size=True)
                applied_degradations.update({"downsample": {"factor": 2, "retain_size": True}})
            elif selected_degrad == 6:
                interpolation = random.randint(0, 6)
                degraded_img = degrad.degradations.resize(degraded_img, 2, retain_size=True, interpolation=interpolation)
                applied_degradations.update({"resize": {"factor": 2, "retain_size": True, "interpolation": interpolation}})
            elif selected_degrad == 7:
                blur_type = blur_types[random.randint(0, len(blur_types)-1)]
                kernel_size = random.randrange(1, 12, 2)
                degraded_img = degrad.degradations.blur(degraded_img, (kernel_size, kernel_size), blur_type)
                applied_degradations.update({"blur": {"kernel_size": kernel_size, "blur_type": blur_type}})


        target_path = f"{output_dset_obj}/{target_img.name}"
        if not cv2.imwrite(target_path, degraded_img):
            logging.warning(f"Could not write {target_path} image.")

        print("Calculating PSNR.")
        psnr_score = quality.calculate.psnr(original_img, degraded_img)
        print(f"PSNR score between original and degraded: {psnr_score}")
        print("Calculating SSIM.")
        ssim_score = quality.calculate.ssim(original_img, degraded_img)
        print(f"SSIM score between original and degraded: {ssim_score}")
        scores.update({str(target_img): {"ssim": ssim_score, "psnr": psnr_score, "degradations": applied_degradations}})
        if inter_num > 100:
            break

    output_json = json.dumps(scores)
    with open("scores.json", "w+") as f:
        f.write(output_json)


