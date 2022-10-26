import unprocessing
import tensorflow
import tensorflow.keras.utils
import cv2
import numpy as np

if __name__ == "__main__":
    img_black = np.zeros([600, 600, 3], dtype=np.float) + 1
    img = cv2.imread("../input.png")
    shot_noise, read_noise = unprocessing.random_noise_levels()
    img = tensorflow.convert_to_tensor(img)
    img /= 254
    noisy = unprocessing.add_noise(img, shot_noise, read_noise, amplification=5).eval(session=tensorflow.compat.v1.Session())
    noisy_black = unprocessing.add_noise(img_black, shot_noise, read_noise, amplification=5).eval(session=tensorflow.compat.v1.Session())
    cv2.imshow("im1", img.eval(session=tensorflow.compat.v1.Session()))
    cv2.imshow("im2", noisy)
    cv2.imshow("im3", noisy_black)
    cv2.waitKey(0)