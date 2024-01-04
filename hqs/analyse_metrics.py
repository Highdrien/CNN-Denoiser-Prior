import os
import json
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt


def analyse(metrics: dict, path) -> None:
    n = len(metrics['x'])
    x = np.array(metrics['x'])
    z = np.array(metrics['z'])
    y = np.array([metrics['y']] * n)
    
    metrics_name = ['MSE', 'PSNR']
    for i in range(len(metrics_name)):
        plt.grid()
        plt.plot(y[:, i])
        plt.plot(x[:, i])
        plt.plot(z[:, i])
        plt.legend(['blured image', 'x_k', 'z_k'])
        # plt.title(f'metric: {metrics_name[i]} sur les itérations k')
        plt.xlabel('itération k')
        plt.ylabel(f'metric: {metrics_name[i]}')
        plt.savefig(os.path.join(path, f'{metrics_name[i]}.png'))
        plt.clf()



if __name__ == '__main__':
    for cnn in ['constant', 'random']:
        path = os.path.join('images', f'hqs_{cnn}')
        with open(file=os.path.join(path, 'metrics.json'), mode='r') as f:
            metrics = json.load(f)
        analyse(metrics=metrics, path=path)