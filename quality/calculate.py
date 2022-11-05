import numpy as np
import math
import logging
import cv2

def calculate_psnr(original_img, modified_img):
    """
    Calculate PSNR between original and modified img.
    :param original_img:
    :param modified_img:
    :return:
    """
    if original_img is None:
        msg = "Original image for PSNR calc cannot be None"
        logging.error(msg)
        raise ValueError(msg)

    if modified_img is None:
        msg = "Target image for PSNR calc cannot be None."
        logging.error(msg)
        raise ValueError(msg)

    gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_modified = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
    maximum_uint8_val = np.iinfo('uint8').max  # get max uint8 val
    mean_square_error = np.power(np.mean(gray_original - gray_modified), 2)
    if mean_square_error == 0:
        return -1  # images are the same
    psnr = 20 * math.log10(maximum_uint8_val / math.sqrt(mean_square_error))
    return np.round(psnr, 4) if psnr > 0 else float('inf')


def __ssim(img1, img2):
    """
    Calculate SSIM for a 11x11 img region.
    :param img1:
    :param img2:
    :return:
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map


def calculate_mssim(original_img, modified_img) -> float:
    """
    Calculate MSSIM between original img modified img. The closer score is to 1, the better img quality.
    SSIM for an RGB is defined as following MSSIM = ||SSIM(R_o, R_m) + SSIM(G_o, G_m) + SSIM(B_o, B_m)||
    Where R,G,B are the channels and _o and _m are original and modified image.
    Similarly to images in different color spaces, you just calculate SSIM for specific channel and take a mean from all.
    channels.
    :param original_img: Original image.
    :param modified_img: Degraded image
    :return: MSSIM val between -1 and 1. The closer to 1, the less degraded image is. Tha closer to -1 the more image is
    'anti-correlated' to the input image.
    """
    if original_img is None:
        msg = "Original image for SSIM calc cannot be None"
        logging.error(msg)
        raise ValueError(msg)

    if modified_img is None:
        msg = "Target image for SSIM calc cannot be None"
        logging.error(msg)
        raise ValueError(msg)
    if original_img.shape != modified_img.shape:
        raise ValueError('Input images must have the same dimensions.')
    ssims = []
    for channel in range(original_img.shape[2]):
        ssims.append(__ssim(original_img[:, :, channel], modified_img[:, :, channel]))
    return np.round(np.array(ssims).mean(), 4)
