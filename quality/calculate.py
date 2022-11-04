import numpy as np
import math
import logging
import cv2
def calculate_psnr(original_img, modified_img):
    if original_img is None:
        msg = "Original image for SSIM calc cannot be None"
        logging.error(msg)
        raise ValueError(msg)

    if modified_img is None:
        msg = "Target image for SSIM calc cannot be None"
        logging.error(msg)
        raise ValueError(msg)

    original_max_pix_val = original_img.max()
    mean_square_error = np.mean(np.power(original_img - modified_img, 2))
    psnr = 20 * math.log10(original_max_pix_val / math.sqrt(mean_square_error))
    return np.round(psnr, 4) if psnr > 0 else float('inf')


def __ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(original_img, modified_img):
    if not original_img.shape == modified_img.shape:
        raise ValueError('Input images must have the same dimensions.')
    if original_img.shape[2] == 3:
        ssims = []
        for i in range(3):
            ssims.append(__ssim(original_img, modified_img))
        return np.round(np.array(ssims).mean(), 4)
    elif original_img.shape[2] == 1:
        return np.round(__ssim(np.squeeze(original_img), np.squeeze(modified_img)))
