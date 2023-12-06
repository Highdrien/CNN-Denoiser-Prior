import os
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


def load_img(param: EasyDict, fileindex: int) -> ndarray:
    """ get the file number <fileindex> """
    filename = os.listdir(param.data.path)[fileindex]
    filepath = os.path.join(param.data.path, filename)
    return np.load(filepath) * 255




if __name__ == '__main__':
    h, H = get_H(param=PARAM)
