import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from torch import Tensor, uint8

from pylops.signalprocessing.convolve2d import Convolve2D


def blurring_gaussian_operator(hdim: Tuple[int, int],
                               hsigma: Tuple[float, float]
                               ) -> np.ndarray:
    hz = np.exp(-hsigma[0] * np.linspace(-(hdim[0] // 2), hdim[0] // 2, hdim[0]) ** 2)
    hx = np.exp(-hsigma[1] * np.linspace(-(hdim[1] // 2), hdim[1] // 2, hdim[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]
    return h


def get_conv2D(image_shape: Tuple[int, int],
               blurring_operator: np.ndarray,
               ) -> Convolve2D:
    hdim = blurring_operator.shape
    Cop = Convolve2D(image_shape[:2], 
                     h=blurring_operator, 
                     offset=(hdim[0] // 2, hdim[1] // 2),
                     dtype="float32")
    return Cop


def blur_image(image: np.ndarray, Cop: Convolve2D) -> np.ndarray:
    return Cop * image


def plot_image_and_blured(image: Tensor, blured_image: Tensor) -> None:
    def pre_process(x: Tensor) -> Tensor:
        if x.dtype != uint8:
            x = x.to(uint8)
        if x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        return x
    
    image = pre_process(image)
    blured_image = pre_process(blured_image)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')

    plt.subplot(1, 2, 2)
    plt.imshow(blured_image)
    plt.title('Blured Image')

    plt.show()
