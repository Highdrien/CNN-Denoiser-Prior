import os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple
from easydict import EasyDict

from pylops.signalprocessing import Convolve2D
from get_parameter import load_parameter


def get_image(image_path: str) -> ndarray:
    return np.array(Image.open(image_path))


def blurring_gaussion_operator(hdim: Tuple[int, int],
                               epsilon: Tuple[float, float]
                               ) -> ndarray:
    hz = np.exp(-epsilon[0] * np.linspace(-(hdim[0] // 2), hdim[0] // 2, hdim[0]) ** 2)
    hx = np.exp(-epsilon[1] * np.linspace(-(hdim[1] // 2), hdim[1] // 2, hdim[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]
    return h


def get_H(h: ndarray, image_shape: Tuple[int, int, int]) -> Convolve2D:
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


def bluring_process(image: ndarray, H: Convolve2D, sigma: int) -> ndarray:
    blured_image =  H * image + generate_noise_vector(shape=image.shape, sigma=sigma)
    return blured_image


def plot_image(original_im: ndarray, degraded_im: ndarray) -> None:
    plt.subplot(1, 2, 1)
    plt.imshow(original_im, vmin=0, vmax=1)
    plt.title("Image originale")

    plt.subplot(1, 2, 2)
    plt.imshow(degraded_im, vmin=0, vmax=1)
    plt.title("Image dégradée")

    plt.show()


def save_blured_images_from_param(param: EasyDict) -> None:
    np.random.seed(0)
    files = os.listdir(param.data.datafrom)[-param.data.num_data:]

    dst_folder = param.data.path
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    h = blurring_gaussion_operator(hdim=param.filter.size,
                                   epsilon=param.filter.epsilon)
    
    H = get_H(h=h, image_shape=param.image_shape)

    for filename in files:
        image = get_image(image_path=os.path.join(param.data.datafrom, filename))
        sigma = np.random.randint(low=1, high=param.noise.sigma)
        blured_image = bluring_process(image=image,H=H, sigma=sigma)
        
        blured_image = blured_image / 255
        dst_path = os.path.join(dst_folder, filename)[:-4]

        np.save(dst_path, blured_image)
        print(f"save image in {dst_path}.npz")  



if __name__ == '__main__':
    # DATA_PATH = os.path.join('cnn', 'data', 'center_patches')
    # image_shape = (256, 256, 3)
    # y = get_image(os.path.join(DATA_PATH, '0002.png'))
    # h = blurring_gaussion_operator(hdim=[5,  5], epsilon=[0.01, 0.01])
    # H = get_H(h=h, image_shape=image_shape)
    # x = bluring_process(image=y, H=H, sigma=10)

    # plot_image(original_im=y, degraded_im=x / 255)

    PARAM = load_parameter(path=os.path.join('hqs', 'parameter.yaml'))
    save_blured_images_from_param(param=PARAM)
