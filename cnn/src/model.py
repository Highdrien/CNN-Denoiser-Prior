import torch.nn as nn
from torch import Tensor, rand, double
from easydict import EasyDict
from numpy import prod

from typing import Optional, List
from icecream import ic


class Block(nn.Module):
    def __init__(self,
                 dilatation: int,
                 image_size: int,
                 in_channels: Optional[int]=64,
                 out_channels: Optional[int]=64,
                 run_batchnorm: Optional[bool]=True,
                 run_relu: Optional[bool]=True,
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding='same',
                              dilation=dilatation)
        self.run_batchnorm = run_batchnorm
        self.run_relu = run_relu
        if self.run_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.run_relu:
            self.run_relu = run_relu
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.run_batchnorm:
            x = self.batchnorm(x)
        if self.run_relu:
            x = self.relu(x)
        return x


class CNN(nn.Module):
    def __init__(self,
                 channels: int,
                 image_size: int,
                 dilitation: Optional[List[int]]=[1, 2, 3, 4, 3, 2, 1]
                 ) -> None:
        super().__init__()

        first_block = Block(dilatation=dilitation[0],
                            image_size=image_size,
                            in_channels=channels,
                            run_batchnorm=False)
        last_block = Block(dilatation=dilitation[-1],
                           image_size=image_size,
                           out_channels=channels,
                           run_batchnorm=False,
                           run_relu=False)
        
        self.blocks = []
        self.blocks.append(first_block)
        for i in dilitation[1:-1]:
            self.blocks.append(Block(dilatation=i,
                                     image_size=image_size))
        self.blocks.append(last_block)

        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x: Tensor) -> Tensor:
        """ 
        x: tensor of shape (channels x image_size x image_size)
        """
        for block in self.blocks:
            x = block(x)
        return x

    def check_param(self) -> None:
        for param in self.parameters():
            ic(param.device, param.dtype)
    
    def get_number_parameters(self) -> int:
        return sum([prod(param.size()) for param in self.parameters()])



def get_model(config: EasyDict) -> CNN:
    model = CNN(channels=3,
                image_size=config.data.image_size,
                dilitation=config.model.dilitation)
    model.to(double)
    return model


def check_device(model: nn.Module) -> None:
    for param in model.parameters():
        ic(param.device, param.dtype)


if __name__ == "__main__":
    model = CNN(channels=3, image_size=16)
    ic(model)
    
    x = rand((64, 3, 16, 16))
    y = model(x)
    print(f"output shape: {y.shape}")
