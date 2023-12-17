import os
import numpy as np
from numpy import ndarray
from easydict import EasyDict
from typing import Tuple
from pylops.signalprocessing import Convolve2D
from PIL import Image
from icecream import ic 
import matplotlib.pyplot as plt

from get_parameter import PARAM
import blured_image
import fourier


def get_H(param: EasyDict) -> Tuple[ndarray, Convolve2D]:
    """ Get H according the parameter """
    h = blured_image.blurring_gaussion_operator(hdim=param.filter.size,
                                                epsilon=param.filter.epsilon)
    H = blured_image.get_H(h=h, image_shape=param.image_shape)
    return (h, H)


def load_img(param: EasyDict, fileindex: int) -> Tuple[ndarray, ndarray]:
    """ get the file number <fileindex> 
    return image y and blured image: x """
    assert 0 <= fileindex <= 9, f"{fileindex} must be between [0, 9]"
    filename = f"0{891 + fileindex}"
    x = np.load(os.path.join(param.data.path, filename + '.npy')) * 255
    y = np.array(Image.open(os.path.join(param.data.path, filename + '.png')))
    return y, x
    




if __name__ == '__main__':
    y, x = load_img(param=PARAM, fileindex=0)
    plt.imshow(x/255)
    plt.show()
    h, H = get_H(param=PARAM)
    modH = fourier.get_module_H_power_2(h=h, image_shape=(256,256,3))
    ic(modH)
    x1 = fourier.find_x(image_shape=(256, 256, 3), mu=1, H=H, h=h, y=y, z=x)
    ic(x1)
    plt.imshow(x1 * 100)
    plt.show()
