import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from PIL.Image import open as Pil_Image_Open
from typing import Tuple
from icecream import ic

from pylops.signalprocessing import Convolve2D


def get_image(image_path: str) -> ndarray:
    return np.array(Pil_Image_Open(image_path))


def blurring_gaussion_operator(hdim: Tuple[int, int],
                               epsilon: Tuple[float, float]
                               ) -> ndarray:
    hz = np.exp(-epsilon[0] * np.linspace(-(hdim[0] // 2), hdim[0] // 2, hdim[0]) ** 2)
    hx = np.exp(-epsilon[1] * np.linspace(-(hdim[1] // 2), hdim[1] // 2, hdim[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]
    return h


def get_H(h: ndarray, image_shape: int) -> Convolve2D:
    """
    image_shape format: (H, W, C)
    """
    hdim = h.shape
    H = Convolve2D(
        image_shape[:-1], h=h, offset=(hdim[0] // 2, hdim[1] // 2), dtype="float32"
    )
    return H


def generate_noise_vector(shape: tuple, sigma: float) -> ndarray:
    return sigma * np.random.randn(*shape)


def bluring_process(image: ndarray, H: Convolve2D, v: ndarray) -> ndarray:
    return H * image + v


def plot_image(original_im: ndarray, degraded_im: ndarray) -> None:
    plt.subplot(1, 2, 1)
    plt.imshow(original_im, vmin=0, vmax=1)
    plt.title("Image originale")

    plt.subplot(1, 2, 2)
    plt.imshow(degraded_im, vmin=0, vmax=1)
    plt.title("Image dégradée")

    plt.show()


if __name__ == '__main__':
    DATA_PATH = os.path.join('..', 'cnn', 'data', 'center_patches')
    image_shape = (256, 256, 3)
    y = get_image(os.path.join(DATA_PATH, '0002.png'))
    h = blurring_gaussion_operator(hdim=[5,  5], epsilon=[0.01, 0.01])
    H = get_H(h=h, image_shape=image_shape)
    x = bluring_process(image=y, H=H, v=generate_noise_vector(image_shape, sigma=0.1))
    print(x)

    plot_image(original_im=y, degraded_im=x / 255)

    # H = np.array(H)
    # ic(H)
    # ic(type(H))
    # ic(H.dtype)
    # ic(H.shape)
    # ic(H.item().shape)
    # ic(H.item().valeurs[0, 0])
    # ic(H[1, 1])
    # H_inv = np.linalg.inv(H)

