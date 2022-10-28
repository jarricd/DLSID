"""Perform series of degradations."""
import argparse
import logging
import sys
from pathlib import Path

import degrad.degradations
import cv2
import degrad.unprocessing

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

    img_files.extend(list(dset_path_obj.glob("*jpg")))
    img_files.extend(list(dset_path_obj.glob("*png")))

    for target_img in img_files:
        loaded_img = cv2.imread(str(target_img))
        intermediate_img = loaded_img.copy()
        pipeline = [degrad.degradations.downsample, degrad.degradations.resize, degrad.unprocessing.add_noise_to_img]
        pipeline_params = [(loaded_img, 2, False), (intermediate_img, 4, cv2.INTER_NEAREST, False), (intermediate_img, 5)
                           ]

        for pipeline_fn, params in zip(pipeline, pipeline_params):
            intermediate_img = pipeline_fn(*params)

        target_path = f"{output_dset_obj}/{target_img.name}"
        if not cv2.imwrite(target_path, intermediate_img):
            logging.warning(f"Could not write {target_path} image.")