import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

from PIL import Image
from easydict import EasyDict
from typing import Optional, Tuple

from torch import nn
from torch import Tensor
from torchvision import transforms


np.random.seed(0)
torch.manual_seed(0)



class Blured:
    """
    generator for train, validation and test
    """
    def __init__(self,
                 data_path: str,
                 noise_variance: float,
                 random_variance: bool,
                 config_name: str,
                 resize: str='random_crop'
                 ) -> None:
        """ create a generator with:
        mode in train, val, test
        data_path: path containing all images
        image size: (c H, W)
        """
        self.data_path = data_path
        self.original_data_path = os.path.join(self.data_path, 'original')
        self.data = list(map(lambda x: os.path.join(self.original_data_path, x), os.listdir(self.original_data_path)))
        self.dst_path = os.path.join(self.data_path, f'{config_name}_blured')
        if not os.path.exists(self.dst_path):
            os.makedirs(self.dst_path)

        self.noise_variance = noise_variance
        self.random_variance = random_variance

        if resize == 'bicubic':
            self.crop_transform = nn.MaxPool2d(kernel_size=4)
        else: 
            self.crop_transform = transforms.RandomCrop(size=(64, 64))
        
    def __len__(self) -> int:
        return len(self.data)

    def blured(self, index: int) -> Tensor:
        """
        get the image and the blured image. both have shape: (c, H, W)
        return: blured_image, image
        """
        # Get image
        image = np.array(Image.open(self.data[index]))
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.from_numpy(image).to(torch.float32)
        image = self.crop_transform(image)

        noise_variance = np.random.randint(1, self.noise_variance) if self.random_variance else self.noise_variance

        blured_image = image + torch.randn_like(image, dtype=torch.float32) * noise_variance
        blured_image = torch.clamp(input=blured_image, min=0, max=255)
        return blured_image / 255
    
    def save_blured_image(self, x: Tensor, filename: str) -> None:
        image_path = os.path.join(self.dst_path, filename)

        x = np.array(x)
        x = np.transpose(x, axes=(1, 2, 0))

        np.save(file=f'{image_path}.npy', arr=x)
        x = (x * 255).astype(np.uint8)
        image = Image.fromarray(x)
        image.save(f'{image_path}.png')
    
    def blured_images(self) -> None:
        filenames = os.listdir(self.original_data_path)
        for i in range(len(self)):
            x = self.blured(index=i)
            self.save_blured_image(x=x, filename=filenames[i][:-4])
        print('done')


def blured_images(config: EasyDict,
                  data_path: str=os.path.join('..', 'images')
                  ) -> None:
    infer = Blured(data_path=data_path,
                   noise_variance=config.data.noise_variance,
                   random_variance=config.data.random_variance,
                   config_name=config.name,
                   resize=config.data.resize)
    infer.blured_images()

 

    