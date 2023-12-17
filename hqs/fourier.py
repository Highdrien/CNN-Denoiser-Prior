import numpy as np
from numpy import ndarray
from easydict import EasyDict
from typing import Tuple
from pylops.signalprocessing import Convolve2D

from get_parameter import PARAM
import blured_image
from icecream import ic 


def get_H(param: EasyDict) -> Tuple[ndarray, Convolve2D]:
    """ Get H according the parameter """
    h = blured_image.blurring_gaussion_operator(hdim=param.filter.size,
                                                epsilon=param.filter.epsilon)
    H = blured_image.get_H(h=h, image_shape=param.image_shape)
    return (h, H)


def get_module_H_power_2(h: ndarray, image_shape: Tuple[int, int, int]) -> float:
    """ trouve le module de H*2 où H = Convolve2D(h) """
    n = image_shape[0] * image_shape[1]
    return n * (np.linalg.norm(h) ** 2)


def find_x_channel(image_shape: Tuple[int, int, int],
                   mu: float,
                   H: Convolve2D,
                   h: ndarray,
                   y: ndarray,
                   z: ndarray):
    """ a * fourier x = fourier b """
    a = get_module_H_power_2(h, image_shape) + mu 
    b = H.transpose() * y + mu * z
    fourier_b = np.fft.fft(b)
    fourier_x = fourier_b / a
    fourier_x = fourier_x.reshape(image_shape[:-1])
    x = np.fft.ifft2(fourier_x)
    return np.real(x)


def find_x(image_shape: Tuple[int, int, int],
           mu: float,
           H: Convolve2D,
           h: ndarray,
           y: ndarray,
           z: ndarray):
    x = np.zeros(shape=image_shape)
    for c in range(image_shape[2]):
        x_c = find_x_channel(image_shape=(256, 256, 3),
                             mu=mu, H=H, h=h,
                             y=y[..., c].flatten(),
                             z=z[..., c].flatten())
    
        x[..., c] = x_c
    
    return x






