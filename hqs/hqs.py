import os
import numpy as np
from numpy import ndarray
from easydict import EasyDict
from typing import Tuple
from pylops.signalprocessing import Convolve2D
from PIL import Image
import matplotlib.pyplot as plt
import sys
import torch
import yaml

from get_parameter import PARAM
import blured_image

sys.path.append(os.path.join(sys.path[0], '..'))
from cnn.src.model import get_model


def get_H(param: EasyDict) -> Tuple[ndarray, Convolve2D]:
    """ Get H according the parameter """
    h = blured_image.blurring_gaussion_operator(hdim=param.filter.size,
                                                epsilon=param.filter.epsilon)
    H = blured_image.get_H(h=h, image_shape=param.image_shape)
    return (h, H)


def load_img(param: EasyDict, fileindex: int) -> Tuple[ndarray, ndarray]:
    """ get the file number <fileindex> 
    return image y and blured image: x """
    path = param.data.blured_path
    filename = os.listdir(path)[fileindex]
    filepath = os.path.join(path, filename)
    blured_image = np.load(file=filepath)
    return blured_image
    

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


def run_hqs_method(param: EasyDict, y: ndarray, num_iter: int, mu: float, save: bool) -> ndarray:
    _, H = get_H(param=PARAM)
    mat_H, inv = inv_H(H, mu)

    model = get_model_param(param=param)

    dst_path = os.path.join(param.data.data_root, f"hqs_{param.cnn.logspath.split('/')[-1]}")
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    
    image = Image.fromarray((y * 255).astype(np.uint8))
    image.save(os.path.join(dst_path, 'y.png'))

    for i in range(num_iter):
        x = find_x(mat_H=mat_H, inv=inv, y=y, mu=mu, z=y)
        z = find_z(x=x, model=model)

        if save:
            image = Image.fromarray((x * 255).astype(np.uint8))
            image.save(os.path.join(dst_path, f'x_{i}.png'))
            image = Image.fromarray((z * 255).astype(np.uint8))
            image.save(os.path.join(dst_path, f'z_{i}.png'))

    return z


def get_model_param(param: EasyDict) -> torch.nn.Module:
    cnn_config_path = os.path.join(param.cnn.logspath, 'config.yaml')
    stream = open(cnn_config_path, 'r')
    cnn_config = EasyDict(yaml.safe_load(stream))
    model = get_model(config=cnn_config)
    checkpoint = torch.load(os.path.join(param.cnn.logspath, 'checkpoint.pt'))
    model.load_state_dict(checkpoint)
    del checkpoint
    return model



if __name__ == '__main__':
    blured_im = load_img(param=PARAM, fileindex=0)

    # plt.imshow(blured_im)
    # plt.show()
    # plt.savefig(os.path.join('debluring', 'x.png'))
    # plt.clf()
    z = run_hqs_method(param=PARAM, y=blured_im, mu=0.1, num_iter=10, save=True)

    # blured_image.plot_image(original_im=blured_im, degraded_im=z, title='Image reconstruite')

    # model = get_model_param(param=PARAM)
    # im_model = torch.tensor(blured_im, requires_grad=False).permute(2, 0, 1).to(torch.float32).unsqueeze(0)
    # with torch.no_grad():
    #     output = model.forward(im_model)
    #     output = output.squeeze(0).permute(1, 2, 0).detach().numpy()
    #     output = np.clip(output, a_min=0, a_max=1)
    # plt.imshow(output)
    # plt.savefig(os.path.join('debluring', 'forward_model.png'))
