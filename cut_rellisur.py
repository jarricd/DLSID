import argparse
import pathlib
import sys
import random
import os
import shutil
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="RELLISUR Cutter",
                                     description="At random select exposure values on RELLISUR dataset.",
                                     epilog="This program does not have supercow powers.")
    parser.add_argument("lllr_path", help="LLLR dataset location.")
    parser.add_argument("output_path", help="Resulting LLLR dset location.")

    args = parser.parse_args()

    lllr_path = pathlib.Path(args.lllr_path)
    output_path = pathlib.Path(args.output_path)

    if not lllr_path.is_dir():
        sys.exit(1)

    if not output_path.is_dir():
        sys.exit(2)

    lllr_images = list(lllr_path.glob("*png"))
    filenames = [f"{img_name.stem}{img_name.suffix}" for img_name in lllr_images]
    numbers = [img_name.stem.split("-")[0] for img_name in lllr_images]
    exposures = [img_name.stem.split("-")[1] for img_name in lllr_images]
    unique_exposures = list(set(exposures))
    unique_numbers = list(set(numbers))
    grouped_filenames = {number: [] for number in numbers}
    for number in unique_numbers:
        for filename in filenames:
            if number in filename:
                grouped_filenames[number].append(filename)

    for key, val in grouped_filenames.items():
        selected_file = random.choice(val)
        print(f"Moving file {selected_file}")
        shutil.move(f"{lllr_path}\{selected_file}", f"{output_path}\{selected_file.split('-')[0]}.png")
