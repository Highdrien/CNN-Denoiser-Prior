import numpy as np
from icecream import ic 
from typing import Tuple
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

def calculate_metrics(image1: np.ndarray, image2: np.ndarray) -> Tuple[float, float]:
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mse = mean_squared_error(image1, image2)

    psnr = peak_signal_noise_ratio(image1, image2)
    return mse, psnr

if __name__ == '__main__':
    x = np.random.random((64, 64, 3))
    y = np.random.random((64, 64, 3))
    calculate_metrics(image1=x, image2=y)

