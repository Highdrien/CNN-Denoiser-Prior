import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PIL_Image
from typing import Tuple, Optional
import os

import pylops


def blurring_gaussion_operator(hdim: Tuple[int, int]) -> np.ndarray:
    hz = np.exp(-0.1 * np.linspace(-(hdim[0] // 2), hdim[0] // 2, hdim[0]) ** 2)
    hx = np.exp(-0.03 * np.linspace(-(hdim[1] // 2), hdim[1] // 2, hdim[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]
    return h


class GrayImage:
    def __init__(self, 
                 image_path: str,
                 hdim: Optional[Tuple[int, int]]=(15, 25)
                 ) -> None:
        """ get image and blured image"""
        self.image = np.array(PIL_Image.open(image_path))
        self.shape = self.image.shape
        print(f"image shape:{self.shape}")
        assert len(self.shape) == 2, f"Error, l'image n'est pas en noir et blanc. shape:{self.shape}"

        self.hdim = hdim
        self.h = blurring_gaussion_operator(hdim=self.hdim)
        self.Cop = self.get_conv2D()
        self.imblur = self.get_blured_image()

    def get_conv2D(self) -> pylops.signalprocessing.convolve2d.Convolve2D:
        Cop = pylops.signalprocessing.Convolve2D(self.shape[:2], 
                                                 h=self.h, 
                                                 offset=(self.hdim[0] // 2, self.hdim[1] // 2),
                                                 dtype="float32")
        return Cop
    
    def get_blured_image(self) -> np.ndarray:
        return self.Cop * self.image
    
    def get_deblured_image(self) -> np.ndarray:
        imdeblur = pylops.optimization.leastsquares.normal_equations_inversion(
            self.Cop, self.imblur.ravel(), None, maxiter=50  # solvers need 1D arrays
        )[0]
        imdeblur = imdeblur.reshape(self.Cop.dims)
        return imdeblur
    
    def get_FISTA_image(self) -> np.ndarray:
        Wop = pylops.signalprocessing.DWT2D(self.shape[:2], wavelet="haar", level=3)

        imdeblurfista = pylops.optimization.sparsity.fista(
            self.Cop * Wop.H, self.imblur.ravel(), eps=1e-1, niter=100
        )[0]
        imdeblurfista = imdeblurfista.reshape((self.Cop * Wop.H).dims)
        imdeblurfista = Wop.H * imdeblurfista
        return imdeblurfista
    
    def get_TV_image(self) -> np.ndarray:
        Dop = [
            pylops.FirstDerivative(self.shape[:2], axis=0, edge=False),
            pylops.FirstDerivative(self.shape[:2], axis=1, edge=False),
        ]

        imdeblurtv = pylops.optimization.sparsity.splitbregman(
            self.Cop,
            self.imblur.ravel(),
            Dop,
            niter_outer=10,
            niter_inner=5,
            mu=1.5,
            epsRL1s=[2e0, 2e0],
            tol=1e-4,
            tau=1.0,
            show=False,
            **dict(iter_lim=5, damp=1e-4)
        )[0]
        imdeblurtv = imdeblurtv.reshape(self.Cop.dims)
        return imdeblurtv
    
    def plot(self) -> None:
        """ plot image """
        plt.imshow(self.image, cmap='gray')
        plt.show()
    
    def plot_blured(self) -> None:
        """ plot the blured image """
        plt.imshow(self.imblur, cmap='gray')
        plt.show()


if __name__ == '__main__':
    # image_path = os.path.join('..', 'debluring', 'python.png')
    image_path = os.path.join('..', 'data', '01.png')
    image = GrayImage(image_path=image_path, hdim=(10, 10))
    image.plot()
    image.plot_blured()
    print(image.get_blured_image())
    # print(image.get_FISTA_image())
    # print(image.get_TV_image())