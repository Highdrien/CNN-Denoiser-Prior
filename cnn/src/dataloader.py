import os
import torch
from torch import Tensor
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Tuple

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
    def __init__(self, mode: str, data_path: str) -> None:
        assert mode in ['train', 'val', 'test'], f"mode must be train, val or test, be mode is {mode}"
        print(f"creation of a dataloader on mode: {mode}")

        self.mode = mode
        self.data_path = data_path

        all_path = os.listdir(self.data_path)
        begin, end = split_data(mode=self.mode, num_data=len(all_path))
        print(f"split: {begin} {end}, and data number: {len(all_path)}")

        get_complete_path = lambda image_name: os.path.join(self.data_path, image_name)
        self.data = list(map(get_complete_path, all_path[begin:end]))
        print(f"number of data in the generator: {len(self)}")


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tensor:
        """
        get the image and the blured image
        """
        image = read_image(self.data[index])
        return image

def create_generator(mode: str, data_path: str, batch_size: int) -> DataLoader:
    """ Returns DataLoader of a generator on mode ('train','val','test')"""
    generator = DataGenerator(mode=mode, data_path=data_path)
    dataloader = DataLoader(generator, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


if __name__ == "__main__":
    generator = DataGenerator(mode='val', data_path=os.path.join('..', 'data', 'center_patches'))
    im = generator.__getitem__(42)
    print(im)
    print(im.shape)
    print(type(im))
