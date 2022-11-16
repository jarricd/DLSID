import argparse
import pathlib
import sys
import os
import logging
import random
import shutil

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Split target directory to Train, Test and Val dsets.",
                                        epilog="This program does not have supercow powers")
    argparser.add_argument("input_path", help="Input dset.")
    argparser.add_argument("output_path", help="Output dset.")
    argparser.add_argument("--ratio", help="Split ratio in form train/test/val")
    
    args = argparser.parse_args()
    input_dset_path = pathlib.Path(args.input_path)
    output_dset_path = pathlib.Path(args.output_path)

    if not input_dset_path.is_dir():
        sys.exit(1)
    
    if not output_dset_path.is_dir():
        sys.exit(2)

    train_path = f"{output_dset_path}/Train"
    test_path = f"{output_dset_path}/Test"
    val_path = f"{output_dset_path}/Val"

    try:
        os.mkdir(test_path)
        os.mkdir(train_path)
        os.mkdir(val_path)
    except FileExistsError:
        logging.info("Output paths exist. Ignoring")

    file_list = list(input_dset_path.glob("*jpg"))
    file_list.extend(input_dset_path.glob("*png"))
    file_count = len(file_list)
    ratio_str = args.ratio.split("/")
    ratios = [int(ratio) for ratio in ratio_str]
    total_ratio = sum(ratios)
    ratio_part = file_count // total_ratio
    train_ratio = ratios[0] * ratio_part
    test_ratio = ratios[1] * ratio_part
    val_ratio = ratios[2] * ratio_part 
    print(train_ratio)
    print(test_ratio)
    print(val_ratio)
    path_lut = {
        0: [train_path, train_ratio],
        1: [test_path, test_ratio],
        2: [val_path, val_ratio]
    }
    for file in file_list:
        file_obj = pathlib.Path(file)
        selected_folder = random.choice(range(0, 3))
        selected_path = path_lut[selected_folder][0]
        remaining = path_lut[selected_folder][1]
        if remaining > 0:
            print(f"Moving file: {file}")
            shutil.move(file, f"{selected_path}/{file.stem}")
            remaining -= 1
            path_lut[selected_folder][1] = remaining
    pass