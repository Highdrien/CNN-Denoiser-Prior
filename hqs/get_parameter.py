import os
import yaml
from easydict import EasyDict


def load_parameter(path: str) -> EasyDict:
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


PARAM = load_parameter(path=os.path.join('hqs', 'parameter.yaml'))