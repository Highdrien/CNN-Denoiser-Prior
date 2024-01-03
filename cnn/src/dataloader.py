import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from easydict import EasyDict
from typing import Optional, Tuple

from torch import nn
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


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
                 noise_variance: float,
                 random_variance: bool,
                 resize: str='random_crop'
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

        self.noise_variance = noise_variance
        self.random_variance = random_variance

        if resize == 'bicubic':
            self.crop_transform = nn.MaxPool2d(kernel_size=4)
        else: 
            self.crop_transform = transforms.RandomCrop(size=(64, 64))


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
        
        # assert image.shape == self.image_size, f"Error, image was load with a wrong shape. Expected: {self.image_size} but found {image.shape}"

        image = torch.from_numpy(image).to(torch.float32)
        image = self.crop_transform(image)

        noise_variance = np.random.randint(1, self.noise_variance) if self.random_variance else self.noise_variance

        blured_image = image + torch.randn_like(image, dtype=torch.float32) * noise_variance
        blured_image = torch.clamp(input=blured_image, min=0, max=255)
        return blured_image / 255, image / 255


def create_generator(mode: str, config: EasyDict) -> DataLoader:
    """ Returns DataLoader of a generator on mode ('train','val','test')"""
    generator = DataGenerator(mode=mode,
                              data_path=config.data.path,
                              image_size=(3, config.data.image_size, config.data.image_size),
                              noise_variance=config.data.noise_variance,
                              random_variance=config.data.random_variance,
                              resize=config.data.resize)
    dataloader = DataLoader(generator,
                            batch_size=config.learning.batch_size,
                            shuffle=config.learning.shuffle,
                            drop_last=config.learning.drop_last)
    return dataloader



def plot_image_and_blured(image: Tensor, blured_image: Tensor) -> None:
    """ plot HR image and the blured image """
    def pre_process(x: Tensor) -> Tensor:
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


if __name__ == "__main__":
    from icecream import ic
    generator = DataGenerator(mode='val',
                              data_path=os.path.join('data', 'all_patches'),
                              image_size=(3, 64, 64),
                              noise_variance=20,
                              random_variance=True,
                              resize='bicubic')
    x, y = generator.__getitem__(42)
    ic(x.shape)
    ic(y.shape)
    plot_image_and_blured(image=y, blured_image=x)
