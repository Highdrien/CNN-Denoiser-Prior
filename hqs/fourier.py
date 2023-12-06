import numpy as np
from numpy import ndarray
from easydict import EasyDict
from typing import Tuple
from pylops.signalprocessing import Convolve2D

from get_parameter import PARAM
import blured_image


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






if __name__ == '__main__':
    h, H = get_H(param=PARAM)
    modH = get_module_H_power_2(h=h, image_shape=(256,256,3))
    print(modH)

