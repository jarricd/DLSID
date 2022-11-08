"""Special thanks to BSRGAN authors for practical degradation pipeline. https://github.com/cszn"""
from random import random
from scipy.linalg import orth
import cv2
import logging
import numpy as np
import random
import torch


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
        downsampled_retained = cv2.pyrUp(downsampled,
                                         dstsize=(factor * downsampled.shape[0], factor * downsampled.shape[1]))
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

    resized = cv2.resize(input_img, (input_img.shape[1] // factor, input_img.shape[0] // factor),
                         interpolation=interpolation)
    if retain_size:
        resized_retained = cv2.resize(resized, (resized.shape[1] * factor, resized.shape[0] * factor),
                                      interpolation=interpolation)
        return resized_retained
    return resized


def blur(input_img, kernel_size, blur_type) -> cv2.Mat:
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
    if input_img[0] is None:
        msg = "Input image for blur cannot be empty."
        logging.error(msg)
        raise ValueError(msg)

    if kernel_size[0] <= 0 or kernel_size[1] <= 0:
        msg = "Kernel size cannot be < 0."
        logging.error(msg)
        raise ValueError(msg)

    input_img = input_img.astype("uint8")
    if blur_type == "avg":
        blurred = cv2.blur(input_img, kernel_size)
    elif blur_type == "gauss":
        sigma_x = random.uniform(0, 10)
        blurred = cv2.GaussianBlur(input_img, kernel_size, sigma_x)
    elif blur_type == "median":
        blurred = cv2.medianBlur(input_img, kernel_size[0])
    elif blur_type == "bilat":
        sigma = random.randint(10, 250)
        blurred = cv2.bilateralFilter(input_img, -1, sigma, kernel_size[0])
    else:
        msg = f"Unknown filter type: {blur_type}"
        logging.error(msg)
        raise ValueError(msg)

    return blurred


def add_poisson_noise(img, amp_factor=1):
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
    poisson_seed = random.random()
    vals = np.power(10, amp_factor * poisson_seed)
    img = (np.random.poisson(img * vals).astype(np.float32) / vals) * 255  # rescale to int again
    img = img.astype(np.uint8)
    return img, poisson_seed


def add_speckle_noise(img, lower_level=2, upper_level=25):
    # speckle noise is like gauss noise except is black and white
    noise_level = random.randint(lower_level, upper_level)
    img = img / 255
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        seed = np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
        img += img * seed
    elif rnum < 0.4:
        seed = np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
        img += img * seed
    else:
        L = upper_level / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        seed = np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
        img += img * seed
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8), rnum, seed


def add_gauss_noise(img, lower_level=2, upper_level=25):
    noise_level = random.randint(lower_level, upper_level)
    img = img / 255
    rnum = np.random.rand()
    if rnum > 0.6:  # add color Gaussian noise
        seed = np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
        img += seed
    elif rnum < 0.4:  # add grayscale Gaussian noise
        seed = np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
        img += seed
    else:  # add  noise
        L = upper_level / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        seed = np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
        img += seed
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8), rnum, seed


def add_jpeg_noise(img):
    quality_factor = random.randint(10, 95)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    return img, quality_factor


def add_webp_noise(img):
    quality_factor = random.randint(10, 95)
    result, encimg = cv2.imencode('.webp', img, [int(cv2.IMWRITE_WEBP_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    return img, quality_factor


def zero_dce_exposure(net, target_img, target_exposure, device, threshold=0.97):
    """
    Lower exposure via CE-ZeroDCE NN.

    :param net:
    :param threshold:
    :param target_img:
    :param target_exposure:
    :return:
    """
    target_img = target_img.astype('float32') / 255.0
    h, w, _ = target_img.shape
    img_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    # 0<=L<=100, -127<=a<=127, -127<=b<=127
    l_channel_t = torch.from_numpy(l_channel).view(1, 1, h, w).to(device)
    l_channel_f = l_channel_t / 100.0
    exp_map = target_exposure * torch.ones_like(l_channel_f)
    stuated_map = (l_channel_f > threshold).int()
    exp_map = exp_map * (1 - stuated_map) + l_channel_f * stuated_map
    with torch.no_grad():
        low_light_l = (net(l_channel_f, exp_map) * 100).squeeze().cpu().detach().numpy()
    torch.cuda.empty_cache()
    scale = low_light_l / (l_channel + 1e-8)
    scale = np.dstack([scale] * 3)
    low_light_img = target_img * scale * 255
    img_out = low_light_img.clip(0, 255).astype('uint8')
    return img_out


def change_brightness(img, gamma) -> cv2.Mat:
    if gamma < 0:
        msg = "Invalid gamma supplied."
        logging.error(msg)
        raise ValueError(msg)

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    output_img = cv2.LUT(img, table)
    output_img = np.clip(output_img, 0.0, 255)
    return output_img
