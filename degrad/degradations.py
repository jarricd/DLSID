"""Special thanks to BSRGAN authors for practical degradation pipeline. https://github.com/cszn"""
from random import random

import cv2
import logging


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

