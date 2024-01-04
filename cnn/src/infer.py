import os
import sys
import torch
import numpy as np
from easydict import EasyDict
from icecream import ic
from os.path import dirname as up
from typing import List
from PIL import Image

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))


from src.model import get_model


def load_blured_image(data_path: str) -> torch.Tensor:
    datas_path = list(map(lambda x: os.path.join(data_path, x), os.listdir(data_path)))
    data = []
    for path in datas_path:
        data.append(np.load(path))
    data = np.array(data)
    data = np.transpose(data, (0, 3, 1, 2))
    data = torch.tensor(data).to(torch.float32)
    return data


def save_results(y: torch.Tensor, dst_path: str, filesname: List[str]) -> None:
    if not os.path.exists(dst_path):
        try:
            os.makedirs(dst_path)
        except OSError as e:
            print(f"Erreur lors de la création du chemin '{dst_path}': {e}")

    y = np.array(y.cpu())
    y = np.transpose(y, (0, 2, 3, 1))
    for i in range(y.shape[0]):
        filename = filesname[i][:-4]
        image_path = os.path.join(dst_path, filename)
        y_i = y[i, ...]
        np.save(file=f'{image_path}.npy', arr=y_i)
        y_i = (y_i * 255).astype(np.uint8)
        image = Image.fromarray(y_i)
        image.save(f'{image_path}.png')
    
    print('infer: done')


def infer(config: EasyDict, logging_path: str, data_path: str) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available() and config.learning.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ic(device)

    # Get data
    x = load_blured_image(os.path.join(data_path, 'blured'))
    x = x.to(device)
    ic(x.shape)

    # Get model
    model = get_model(config)
    model = model.to(device)
    checkpoint_path = os.path.join(logging_path, 'checkpoint.pt')
    assert os.path.isfile(checkpoint_path), f'Error: model weight was not found in {checkpoint_path}'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ic(model)

    with torch.no_grad():
        model.eval()
        y = model.forward(x)
        y = y.clamp(min=0, max=1)
    ic(y.shape)

    save_results(y=y,
                 dst_path=os.path.join(data_path, f'infer_{config.name}'),
                 filesname=os.listdir(os.path.join(data_path, 'blured')))
    

if __name__ == '__main__':
    data_path = os.path.join('..', 'images', 'blured')
    # x = load_blured_image(data_path)
    # ic(x.dtype, x.shape)
    # device = torch.device("cuda")

    # y = torch.rand((12, 3, 64, 64), device=torch.device("cuda"))
    # save_results(y, dst_path=os.path.join('..', 'images', 'random'), filesname=os.listdir(data_path))