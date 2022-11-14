import argparse
import logging
import cv2
import degrad.handcrafted_degradations
import random
import sys
import torch
import quality.calculate
import json
import os
from pathlib import Path

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Remove non-square files",
                                         epilog="This program does not have supercow powers.")
    arg_parser.add_argument("dset_path", help="Path of input dataset.")
    args = arg_parser.parse_args()

    dset_path_obj = Path(args.dset_path)

    if not dset_path_obj.is_dir():
       logging.error("Provided dset path is not a directory. Exiting.")
       sys.exit(1)

    img_files = []

    img_files.extend(list(dset_path_obj.glob("*jpg")))

    for target_img in img_files:
        degraded_img = cv2.imread(str(target_img))
        print(f"File {target_img} res {degraded_img.shape[0]} x {degraded_img.shape[1]}")
        if degraded_img.shape[0] != degraded_img.shape[1]:
                print(f"Removing file {target_img}")
                os.remove(str(target_img))
