"""
Class copied from:
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""

import numpy as np
import copy

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.epoch = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model):
        self.epoch += 1
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = self.epoch
            self.counter = 0
            self.best_model = copy.deepcopy(model)

    def __str__(self):
        return "EarlyStopping ( \n patience: {0},\n delta: {1} \n )".format(self.patience, self.delta)