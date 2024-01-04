import os
import yaml
import argparse
from easydict import EasyDict
from icecream import ic
from typing import Optional

from src.train import train
from src.test import test
from src.infer import infer
from src.dataloader import create_generator, plot_image_and_blured


def load_config(path: Optional[str]='config/config.yaml') -> EasyDict:
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


def find_config(experiment_path: str) -> str:
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml', os.listdir(experiment_path)))

    if len(yaml_in_path) == 1:
        return os.path.join(experiment_path, yaml_in_path[0])

    if len(yaml_in_path) == 0:
        print("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        print("ERROR: a lot a .yaml was found in", experiment_path)
    
    exit()

IMPLEMENTED = ['train', 'data', 'test', 'infer']

def main(options: dict) -> None:

    assert options['mode'] in IMPLEMENTED, f"Error, expected mode must in {IMPLEMENTED} but found {options['mode']}"

    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        ic(config)
        train(config)
    
    if options['mode'] == 'data':
        config = load_config(options['config_path'])
        generator = create_generator(mode='val', config=config)
        print(f"len(val_generator): {len(generator)}")
        for x, y in generator:
            x, y = x[0], y[0]
            plot_image_and_blured(image=y, blured_image=x)
            break  
    
    if options['mode'] == 'test':
        assert options['path'] is not None, 'Error, please enter the path of your experimentation that you want to test'
        config_path = find_config(experiment_path=options['path'])
        config = load_config(config_path)
        ic(config)
        test(config=config, logging_path=options['path'])
    
    if options['mode'] == 'infer':
        assert options['path'] is not None, 'Error, please enter the path of your experimentation that you want to test'

        config_path = find_config(experiment_path=options['path'])
        config = load_config(config_path)
        ic(config)
        infer(config=config, logging_path=options['path'], data_path=options['data_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str,
                        help="choose a mode between 'train', 'data'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str,
                        help="path to config (for training)")
    parser.add_argument('--path', type=str,
                        help="experiment path (for test, prediction or generate)")
    parser.add_argument('--data_path', type=str, default=os.path.join('..', 'images'),
                        help='path to the data for the inference: must containt a blured folder')
    args = parser.parse_args()
    options = vars(args)

    main(options)