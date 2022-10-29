"""Special thanks to BSRGAN authors for practical degradation pipeline. https://github.com/cszn"""
from random import random

import cv2
import logging
import tensorflow as tf

def downsample(input_img, factor, retain_size=True) -> cv2.Mat:
    """
    Downsample target image.

    :param input_img: Input img path
    :param factor: Divide image size by factor
    :param retain_size: Flag controlling if we should retain image size.
    :return: Downsampled image
    """
    if input_img is None:
        msg = "Input image for downsampling cannot be empty."
        logging.error(msg)
        raise ValueError(msg)

    downsampled = cv2.pyrDown(input_img, dstsize=(input_img.shape[1] // factor, input_img.shape[0] // factor))
    if retain_size:
        downsampled_retained = cv2.pyrUp(downsampled, dstsize=(factor * downsampled.shape[0], factor * downsampled.shape[1]))
        return downsampled_retained
    return downsampled


def resize(input_img, factor, interpolation=cv2.INTER_NEAREST, retain_size=True) -> cv2.Mat:
    """
    Resize image to target size.

    :param input_img: Input img.
    :param interpolation: Target interpolation
    :param factor: Resize factor.
    :param retain_size: Flag controlling if we should retain image size.
    :return: Resized image
    """
    if input_img is None:
        msg = "Input image for resize cannot be empty."
        logging.error(msg)
        raise ValueError(msg)

    resized = cv2.resize(input_img, (input_img.shape[1] // factor, input_img.shape[0] // factor), interpolation=interpolation)
    if retain_size:
        resized_retained = cv2.resize(resized, (resized.shape[1] * factor, resized.shape[0] * factor),
                             interpolation=interpolation)
        return resized_retained
    return resized


def blur(*args) -> cv2.Mat:
    """
    Blur image.
    NOTE: Median blur accepts only a square kernel, only x from kernel_size tuple will be considered.
    Despite args being passed as *args, positions make a difference and

    :param input_img: Input img.
    :param kernel_size: Kernel size in form of a tuple (x, y)
    :param blur_type: The type of blur.
    :param rest: Refer to specific blur docs, stuff like sigma
    :return: Blurred image.
    """
    if args[0] is None:
        msg = "Input image for blur cannot be empty."
        logging.error(msg)
        raise ValueError(msg)

    if args[1][0] <= 0 or args[1][1] <= 0:
        msg = "Kernel size cannot be < 0."
        logging.error(msg)
        raise ValueError(msg)

    blur_lut = {"block": cv2.blur, "gauss": cv2.GaussianBlur, "median": cv2.medianBlur, "bilat": cv2.bilateralFilter}
    try:
        blur_fn = blur_lut[args[2]]
    except KeyError as e:
        msg = f"{args[2]} is not a valid blur. Valid choices are {list(blur_lut.keys())}."
        logging.error(msg)
        raise Exception(msg) from e

    # reconstruct args without the blur type string
    reconstructed_args = tuple((item for item in args if type(item) is not str))
    blurred = blur_fn(*reconstructed_args)
    return blurred


def reverse_isp_add_noise(input_img) -> cv2.Mat:
    """Add noise to an image using reverse ISP pipeline and reprocess the image to RGB again."""
    if input_img is None:
        msg = "Input image for blur cannot be empty."
        logging.error(msg)
        raise ValueError(msg)

