import os
import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict

from typing import Optional, Tuple
from src.utils import blurring_gaussian_operator, get_conv2D, plot_image_and_blured

np.random.seed(0)
torch.manual_seed(0)


def split_data(mode: str,
               num_data: int,
               split_factor: Optional[Tuple[float, float]]=(0.6, 0.8)
               ) -> Tuple[int, int]:
    split_1 = int(split_factor[0] * num_data)
    split_2 = int(split_factor[1] * num_data)

    if mode == 'train':
        return (0, split_1)
    if mode == 'val':
        return (split_1, split_2)
    if mode == 'test':
        return (split_1, num_data)


class DataGenerator(Dataset):
    """
    generator for train, validation and test
    """
    def __init__(self,
                 mode: str,
                 data_path: str,
                 image_size: Tuple[int, int, int],
                 hstd: Optional[float]=0.1,
                 noise: Optional[float]=0.02
                 ) -> None:
        """ create a generator with:
        mode in train, val, test
        data_path: path containing all images
        image size: (c H, W)
        """
        assert mode in ['train', 'val', 'test'], f"mode must be train, val or test, be mode is {mode}"
        print(f"creation of a dataloader on mode: {mode}")

        self.mode = mode
        self.data_path = data_path
        self.image_size = image_size

        all_path = os.listdir(self.data_path)
        begin, end = split_data(mode=self.mode, num_data=len(all_path))
        print(f"split: {begin} {end}, and data number: {len(all_path)}")

        get_complete_path = lambda image_name: os.path.join(self.data_path, image_name)
        self.data = list(map(get_complete_path, all_path[begin:end]))
        print(f"number of data in the generator: {len(self)}")

        self.hdim_range = [2, 4, 6, 8, 10]
        self.generate_hdim = lambda : np.random.choice(self.hdim_range)
        self.generate_hsigma = lambda : abs(np.random.normal(loc=0, scale=hstd))
        self.duplicate = lambda x: (x, x)
        self.noise = noise


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        get the image and the blured image. both have shape: (c, H, W)
        return: blured_image, image
        """
        # Get image
        image = np.array(Image.open(self.data[index]))
        image = np.transpose(image, (2, 0, 1))
        assert image.shape == self.image_size, f"Error, image was load with a wrong shape. Expected: {self.image_size} but found {image.shape}"

        # Generate blurring gaussian operator
        h = blurring_gaussian_operator(hdim=self.duplicate(self.generate_hdim()),
                                       hsigma=self.duplicate(self.generate_hsigma()))
        cop = get_conv2D(image_shape=self.image_size[1:], blurring_operator=h)

        # blur image
        blured_image = []
        for channel in range(self.image_size[0]):
            blured_image.append(cop * image[channel])
        blured_image = np.array(blured_image)

        image = torch.from_numpy(image).to(torch.float64)
        blured_image = torch.from_numpy(blured_image) + torch.randn(*self.image_size) * self.noise

        return blured_image, image

def create_generator(mode: str, config: EasyDict) -> DataLoader:
    """ Returns DataLoader of a generator on mode ('train','val','test')"""
    generator = DataGenerator(mode=mode,
                              data_path=config.data.path,
                              image_size=(3, config.data.image_size, config.data.image_size))
    dataloader = DataLoader(generator,
                            batch_size=config.learning.batch_size,
                            shuffle=config.learning.shuffle,
                            drop_last=config.learning.drop_last)
    return dataloader


if __name__ == "__main__":
    generator = DataGenerator(mode='val', data_path=os.path.join('..', 'data', 'center_patches'), image_size=(3, 256, 256))
    x, y = generator.__getitem__(42)
    plot_image_and_blured(image=y, blured_image=x)
