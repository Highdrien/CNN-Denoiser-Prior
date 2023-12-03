import os
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from icecream import ic

from src.dataloader import create_generator
from src.model import get_model
from src.metrics import Metrics
from config.config import test_logger


def test(config: EasyDict, logging_path: str) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available() and config.learning.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ic(device)

    # Get data
    test_generator = create_generator(config=config, mode='test')
    n_test = len(test_generator)
    ic(n_test)

    # Get model
    model = get_model(config)
    model = model.to(device)
    checkpoint_path = os.path.join(logging_path, 'checkpoint.pt')
    assert os.path.isfile(checkpoint_path), f'Error: model weight was not found in {checkpoint_path}'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ic(model)
    
    # Loss
    assert config.learning.loss == 'mse', NotImplementedError
    criterion = torch.nn.MSELoss(reduction='mean')

    metrics = Metrics(config=config.metrics, device=device)
    num_metrics = metrics.num_metrics

    ###############################################################
    # Start Evaluation                                            #
    ###############################################################
    test_loss = 0
    test_range = tqdm(test_generator)
    test_metrics = np.zeros(num_metrics)

    with torch.no_grad():
        for x, y_true in test_range:
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model.forward(x)
                
            loss = criterion(y_pred, y_true)

            
            test_loss += loss.item()
            test_metrics += metrics.compute(y_pred=y_pred, y_true=y_true)

            test_range.set_description(f"TEST -> loss: {loss.item():.4f}")
            test_range.refresh()

    ###################################################################
    # Save Scores in logs                                             #
    ###################################################################
    test_loss = test_loss / n_test
    test_metrics = test_metrics / n_test
    
    test_logger(path=logging_path,
                metrics=['mse'] + metrics.metrics_name,
                values=[test_loss] + list(test_metrics))
