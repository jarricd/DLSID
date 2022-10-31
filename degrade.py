"""Perform series of degradations."""
import argparse
import logging
import pathlib
import sys
from pathlib import Path

import degrad.degradations
import cv2
import degrad.unprocessing
import degrad.process
import tensorflow as tf
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate LLLR images out of target dataset.",
                                         epilog="This program does not have supercow powers.")
    arg_parser.add_argument("dset_path", help="Path of input dataset.")
    arg_parser.add_argument("output_path", help="Path of output dataset.")
    arg_parser.add_argument("--noise_amp", help="Amplify noise by supplied factor.", default=1)
    arg_parser.add_argument("--same_degrad", help="Use the degradation pattern for every image.", default=False)
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

    # img_files.extend(list(dset_path_obj.glob("*jpg")))
    # img_files.extend(list(dset_path_obj.glob("*png")))
    img_files = [pathlib.Path("00001.png")]
    loaded_img = cv2.imread(str(img_files[0]))
    original = loaded_img.copy()
    # loaded_img = degrad.degradations.add_poisson_noise(loaded_img)
    # loaded_img = degrad.degradations.add_speckle_noise(loaded_img, 2, 25)
    # loaded_img = degrad.degradations.add_gauss_noise(loaded_img, 2, 25)
    loaded_img_jpeg = degrad.degradations.add_jpeg_noise(loaded_img.copy())
    loaded_img = degrad.degradations.add_webp_noise(loaded_img)
    cv2.imshow("img", loaded_img)
    cv2.imshow("img1", loaded_img_jpeg)
    cv2.imshow("img2", original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for target_img in img_files:
    #     pipeline = []
    #     pipeline_params = []
    #
    #
    #
    #     target_path = f"{output_dset_obj}/{target_img.name}"
    #     if not cv2.imwrite(target_path, output_img):
    #         logging.warning(f"Could not write {target_path} image.")