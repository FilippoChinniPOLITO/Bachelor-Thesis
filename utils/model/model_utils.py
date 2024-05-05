import torch
from torch import nn
from experiments.weed_mapping_experiment.backend.loss.focal_loss import FocalLoss
from experiments.weed_mapping_experiment.backend.model.lawin_model import Lawin
from experiments.MNIST_experiment.backend.model.MLP import MLP


def init_model(model_str: str, extra_args: dict):
    if model_str == 'MLP':
        mlp_architecture = extra_args['network_architecture']
        mlp_activation = extra_args['activation']
        return MLP(layer_sizes=mlp_architecture, activation=mlp_activation)
    elif model_str == 'Lawin':
        return Lawin(extra_args)


def get_activation_fn(activation_str: str):
    if activation_str == 'sigmoid':
        return nn.Sigmoid()
    elif activation_str == 'relu':
        return nn.ReLU()
    elif activation_str == 'tanh':
        return nn.Tanh()


def get_loss_fn(loss_str: str, extra_args: dict = None):
    if loss_str == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif loss_str == 'Focal':
        if (extra_args is None) or ('gamma' not in extra_args) or ('weight' not in extra_args):
            return FocalLoss()
        focal_loss = build_focal_loss(gamma=extra_args['gamma'], weight=extra_args['weight'])
        return focal_loss


def build_focal_loss(gamma: float, weight: any):
    return FocalLoss(gamma=gamma, weight=weight)


def get_optimizer(model, optimizer_str: str, learning_rate: float):
    if optimizer_str == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_str == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)