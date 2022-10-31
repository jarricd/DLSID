"""Special thanks to BSRGAN authors for practical degradation pipeline. https://github.com/cszn"""
from random import random
from scipy.linalg import orth

import cv2
import logging
import numpy as np
import random

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


def add_poisson_noise(img):
    """
    Add poisson noise to img.

    :param img:
    :return:
    """
    if img is None:
        msg = "Input image for noise cannot be empty."
        logging.error(msg)
        raise ValueError(msg)

    img = img / 255  # convert to float
    vals = np.power(10, (2*random.random()+2.0))
    img = (np.random.poisson(img * vals).astype(np.float32) / vals) * 255  # rescale to int again
    img = img.astype(np.uint8)
    return img


def add_speckle_noise(img,  lower_level=2, upper_level=25):
    # speckle noise is like gauss noise except is black and white
    noise_level = random.randint(lower_level, upper_level)
    img = img / 255
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = upper_level/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img*np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def add_gauss_noise(img, lower_level=2, upper_level=25):
    noise_level = random.randint(lower_level, upper_level)
    img = img / 255
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = upper_level/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)

def add_jpeg_noise(img):
    quality_factor = random.randint(10, 95)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    return img

def add_webp_noise(img):
    quality_factor = random.randint(10, 95)
    result, encimg = cv2.imencode('.webp', img, [int(cv2.IMWRITE_WEBP_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    return img


