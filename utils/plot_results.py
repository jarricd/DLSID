import json
import matplotlib.pyplot as plt
import argparse
import pathlib
import sys
import numpy as np

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Parse JSON from lednet log")
    argparser.add_argument("input_path", help="Input log")
    args = argparser.parse_args()

    log_path_obj = pathlib.Path(args.input_path)

    if not log_path_obj.is_file():
        sys.exit(1)

    with open(log_path_obj, "r") as f:
        json_obj = json.loads(f.read())

    x_iter_vals = [int(x) for x in list(json_obj.keys())]
    lr_vals = []
    l_pix = []
    l_side_pix = []
    l_percep = []
    l_side_percep = []
    for key, val in json_obj.items():
        lr_vals.append(val["lr"])
        l_pix.append(val["l_pix"])
        l_side_pix.append(val["l_side_pix"])
        l_percep.append(val["l_percep"])
        l_side_percep.append(val["l_side_percep"])

    plt.figure(figsize=(30, 10))
    plt.plot(x_iter_vals, lr_vals)
    plt.xticks(np.arange(0, max(x_iter_vals) + 1, 10000))
    plt.suptitle("Learning rate change over iterations")
    plt.xlabel("Iteration num")
    plt.ylabel("Learning rate")
    plt.show()

    plt.figure(figsize=(30, 10))
    plt.plot(x_iter_vals, l_pix)
    plt.xticks(np.arange(0, max(l_pix) + 1, 10000))
    plt.suptitle("Pix loss over iterations")
    plt.xlabel("Iteration num")
    plt.ylabel("Pix loss")
    plt.show()

    plt.figure(figsize=(30, 10))
    plt.plot(x_iter_vals, l_side_pix)
    plt.xticks(np.arange(0, max(l_side_pix) + 1, 10000))
    plt.suptitle("Side pix loss over iterations")
    plt.xlabel("Iteration num")
    plt.ylabel("Side pix loss")
    plt.show()

    plt.figure(figsize=(30, 10))
    plt.plot(x_iter_vals, l_percep)
    plt.xticks(np.arange(0, max(l_percep) + 1, 10000))
    plt.suptitle("Percep loss over iterations")
    plt.xlabel("Iteration num")
    plt.ylabel("Percep loss")
    plt.show()

    plt.figure(figsize=(30, 10))
    plt.plot(x_iter_vals, l_side_percep)
    plt.xticks(np.arange(0, max(l_side_percep) + 1, 10000))
    plt.suptitle("Side percep loss over iterations")
    plt.xlabel("Iteration num")
    plt.ylabel("Side percep loss")
    plt.show()

    sys.exit(0)