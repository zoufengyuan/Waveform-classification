import torch
import numpy as np
import time, os
from sklearn.metrics import f1_score,accuracy_score
from torch import nn


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre),accuracy_score(y_true, y_pre)


def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Loss_cal(nn.Module):
    def __init__(self):
        super(Loss_cal, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = 0.5
        self.gamma = 2
        
    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt)**self.gamma * loss
        return torch.mean(F_loss)
