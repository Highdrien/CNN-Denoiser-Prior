import os
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from torch import Tensor, uint8

from pylops.signalprocessing.convolve2d import Convolve2D


def blurring_gaussian_operator(hdim: Tuple[int, int],
                               hsigma: Tuple[float, float]
                               ) -> np.ndarray:
    """ get a blurring gaussian operator """
    hz = np.exp(-hsigma[0] * np.linspace(-(hdim[0] // 2), hdim[0] // 2, hdim[0]) ** 2)
    hx = np.exp(-hsigma[1] * np.linspace(-(hdim[1] // 2), hdim[1] // 2, hdim[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]
    return h


def get_conv2D(image_shape: Tuple[int, int],
               blurring_operator: np.ndarray,
               ) -> Convolve2D:
    """ get a matric H such that shape(H)=shape(image) """
    hdim = blurring_operator.shape
    Cop = Convolve2D(image_shape[:2], 
                     h=blurring_operator, 
                     offset=(hdim[0] // 2, hdim[1] // 2),
                     dtype="float32")
    return Cop


def blur_image(image: np.ndarray, Cop: Convolve2D) -> np.ndarray:
    return Cop * image


def plot_image_and_blured(image: Tensor, blured_image: Tensor) -> None:
    """ plot HR image and the blured image """
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


def print_loss_and_metrics(train_loss: float,
                           val_loss: float,
                           metrics_name: List[str],
                           train_metrics: List[float],
                           val_metrics: List[float]) -> None:
    """ print loss and metrics for train and validation """
    print(f"{train_loss = }")
    print(f"{val_loss = }")
    for i in range(len(metrics_name) - 1):
        print(f"{metrics_name[i]} -> train: {train_metrics[i]:.3f}   val:{val_metrics[i]:.3f}")
    print(f"{metrics_name[-1]} -> train: {np.exp(train_metrics[-1]):.2e}   val:{np.exp(val_metrics[-1]):.2e}")


def save_learning_curves(path: str) -> None:
    result, names = get_result(path)

    epochs = result[:, 0]
    for i in range(1, len(names), 2):
        train_metrics = result[:, i]
        val_metrics = result[:, i + 1]
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title(names[i])
        plt.xlabel('epoch')
        plt.ylabel(names[i])
        plt.legend(names[i:])
        plt.grid()
        plt.savefig(os.path.join(path, names[i] + '.png'))
        plt.close()


def get_result(path: str) -> Tuple[List[float], List[str]]:
    with open(os.path.join(path, 'train_log.csv'), 'r') as f:
        names = f.readline()[:-1].split(',')
        result = []
        for line in f:
            result.append(line[:-1].split(','))

        result = np.array(result, dtype=float)
    f.close()
    return result, names