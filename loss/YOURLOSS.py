import torch.nn as nn

def init_loss_func(args):
    return nn.CrossEntropyLoss()
