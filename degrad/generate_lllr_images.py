import pathlib
import logging
import unprocessing
import tensorflow
import tensorflow.keras.utils
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate LLLR images out of target dataset.",
                                         epilog="This program does not have supercow powers.")
    arg_parser.add_argument("dset_path", help="Path of input dataset.")
    arg_parser.add_argument("output_path", help="Path of output dataset.")
    arg_parser.add_argument("--noise_amp", help="Amplify noise by supplied factor.", default=1)
    arg_parser.add_argument("--same_noise", help="Use the same noise pattern for every image.", default=False)
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

    if args.same_noise:
        shot_noise, read_noise = unprocessing.random_noise_levels()

    for target_img in img_files:
        logging.info(f"Adding noise to the following file: {target_img.stem}")
        img = cv2.imread(str(target_img))
        if img is None:
            logging.error("Loaded image is empty. Ensure path is correct.")
            sys.exit(3)

        if not args.same_noise:
            shot_noise, read_noise = unprocessing.random_noise_levels()

        img = tensorflow.convert_to_tensor(img)
        img /= 254
        noisy = unprocessing.add_noise(img, shot_noise, read_noise,
                                       amplification=args.noise_amp)
        noisy *= 254
        target_path = f"{output_dset_obj}/{target_img.name}"
        if not cv2.imwrite(target_path, noisy.eval(session=tensorflow.compat.v1.Session())):
            logging.warning(f"Could not write {target_path} image.")

    sys.exit(0)