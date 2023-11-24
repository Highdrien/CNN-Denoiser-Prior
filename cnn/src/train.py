import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
# import numpy as np
import time
from easydict import EasyDict
from icecream import ic

from src.dataloader import create_generator
from src.model import get_model, check_device
from src.utils import print_loss_and_metrics, save_learning_curves
from config.config import train_logger, train_step_logger


def train(config: EasyDict) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available() and config.learning.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ic(device)

    # Get data
    train_generator = create_generator(config=config, mode='train')
    val_generator = create_generator(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator)
    ic(n_train, n_val)

    # Get model
    model = get_model(config)
    model = model.to(device)
    ic(model)
    ic(model.get_number_parameters())
    
    # Loss
    assert config.learning.loss == 'crossentropy', NotImplementedError
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Optimizer and Scheduler
    assert config.learning.optimizer == 'adam', NotImplementedError
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=config.learning.milesstone, gamma=config.learning.gamma)


    save_experiment = config.save_experiment
    ic(save_experiment)
    if save_experiment:
        logging_path = train_logger(config)
        best_val_loss = 10e6

    # Metrics
    # metrics_name = list(filter(lambda x: config.metrics[x] is not None, config.metrics))

    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        ic(epoch)
        train_loss = 0
        train_range = tqdm(train_generator)
        # train_metrics = np.zeros(len(metrics_name))

        # Training
        for x, y_true in train_range:
            
            # ic(device)
            x = x.to(device)
            y_true = y_true.to(device)
            # ic(x.shape, x.device, x.dtype)
            # ic(y_true.shape, y_true.device, y_true.dtype)

            y_pred = model.forward(x)

            # ic(y_pred.shape, y_pred.device, y_pred.dtype)
                
            loss = criterion(y_pred, y_true)

            y_pred = torch.nn.functional.softmax(y_pred, dim=1)

            train_loss += loss.item()
            # train_metrics += compute_metrics(config, y_true, y_pred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {loss.item():.4f}")
            train_range.refresh()


        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_range = tqdm(val_generator)
        # val_metrics = np.zeros(len(metrics_name))

        with torch.no_grad():
            
            for x, y_true in val_range:
                x.to(device)
                y_true.to(device)

                y_pred = model.forward(x)
                    
                loss = criterion(y_pred, y_true)

                y_pred = torch.nn.functional.softmax(y_pred, dim=1)
                
                val_loss += loss.item()
                # val_metrics += compute_metrics(config, y_true, y_pred)

                val_range.set_description(f"VAL -> epoch: {epoch} || loss: {loss.item():.4f}")
                val_range.refresh()
        
        scheduler.step()

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        # train_metrics = train_metrics / n_train
        # val_metrics = val_metrics / n_val
        
        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss, 
                              train_metrics=[], 
                              val_metrics=[])
            
            if config.model.save_checkpoint != False and val_loss < best_val_loss:
                print('save model weights')
                model.save(logging_path)
                best_val_loss = val_loss

        print_loss_and_metrics(train_loss=train_loss,
                               val_loss=val_loss,
                               metrics_name=[],
                               train_metrics=[],
                               val_metrics=[])        


    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")

    if save_experiment and config.learning.save_learning_curves:
        save_learning_curves(logging_path)


