import os
import numpy as np
from numpy import ndarray
from easydict import EasyDict
from typing import Tuple
from pylops.signalprocessing import Convolve2D
from PIL import Image
from icecream import ic 
import matplotlib.pyplot as plt
import sys
import torch
import yaml

from get_parameter import PARAM
import blured_image

sys.path.append(os.path.join(sys.path[0], '..'))
# print(sys.path)
from cnn.src.model import get_model


def get_H(param: EasyDict) -> Tuple[ndarray, Convolve2D]:
    """ Get H according the parameter """
    h = blured_image.blurring_gaussion_operator(hdim=param.filter.size,
                                                epsilon=param.filter.epsilon)
    H = blured_image.get_H(h=h, image_shape=param.image_shape)
    return (h, H)


def load_img(fileindex: int) -> Tuple[ndarray, ndarray]:
    """ get the file number <fileindex> 
    return image y and blured image: x """
    assert 0 <= fileindex <= 9, f"{fileindex} must be between [0, 9]"
    filename = f"0{891 + fileindex}"
    x = np.load(os.path.join('blured_data', filename + '_real.npy'))
    y = np.load(os.path.join('blured_data', filename + '_blured.npy'))
    ic(x.shape, y.shape)
    return y, x
    

def inv_H(H: Convolve2D, mu: float) -> Tuple[ndarray, ndarray]:
    """ renvoie la matrice H et (H^T * H+muI)^-1"""
    print('calcule de mat_H')
    mat_H = H.todense()
    print('calcule de la multiplication')
    mat_to_inv = mat_H.T @ mat_H + mu * np.identity(mat_H.shape[0])
    print('inversion')
    inv = np.linalg.inv(mat_to_inv)
    return mat_H, inv


def find_x(mat_H: ndarray, inv: ndarray, y: ndarray, mu: float, z: ndarray) -> ndarray:
    """
    shape y, z -> (H, W, C)
    """
    x = np.zeros_like(y)
    ic(x.shape)
    for c in range(x.shape[-1]):
        xc = inv @ (mat_H @ y[..., c].flatten() + mu * z[..., c].flatten())
        x[..., c] = xc.reshape((64, 64))
    x = np.clip(x, a_min=0, a_max=1)
    return x


def find_z(x: ndarray, model: torch.nn.Module) -> ndarray:
    """
    x: ndarray with shape: (H, W, C)
    z: ndarray with shape: (H, W, C)
    """
    x = torch.tensor(x, requires_grad=False).permute(2, 0, 1).to(torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        z = model.forward(x)
    z = z.squeeze(0).permute(1, 2, 0).detach().numpy()
    z = np.clip(z, a_min=0, a_max=1)
    return z


def run_hqs_method(param: EasyDict, y: ndarray, num_iter: int, mu: float, plot: bool):
    _, H = get_H(param=PARAM)
    mat_H, inv = inv_H(H, mu)

    cnn_config_path = os.path.join(param.cnn.logspath, 'config.yaml')
    stream = open(cnn_config_path, 'r')
    cnn_config = EasyDict(yaml.safe_load(stream))
    ic(cnn_config)
    model = get_model(config=cnn_config)
    checkpoint = torch.load(os.path.join(param.cnn.logspath, 'checkpoint.pt'))
    model.load_state_dict(checkpoint)
    del checkpoint

    image_shape = tuple(param.image_shape)
    print(image_shape)

    dst_path = 'debluring'
    plt.imshow(y)
    plt.savefig(os.path.join(dst_path, 'y.png'))
    plt.clf()

    for i in range(num_iter):
        x = find_x(mat_H=mat_H, inv=inv, y=y, mu=mu, z=y)
        z = find_z(x=x, model=model)

        if plot:
            plt.imshow(x)
            plt.savefig(os.path.join(dst_path, f'x_{i}.png'))
            plt.clf()
            plt.imshow(z)
            plt.savefig(os.path.join(dst_path, f'z_{i}.png'))
            plt.clf()

    return z


if __name__ == '__main__':
    y, x = load_img(fileindex=0)

    plt.imshow(x)
    plt.savefig(os.path.join('debluring', 'x.png'))
    plt.clf()

    z = run_hqs_method(param=PARAM, y=y, mu=0.1, num_iter=3, plot=True)

    blured_image.plot_image(original_im=x, degraded_im=z)
