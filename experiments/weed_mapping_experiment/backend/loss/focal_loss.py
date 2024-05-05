import torch
import torch.nn.functional as F
from torch.nn import Module


class FocalLoss(Module):
    def __init__(self, gamma: float = 0, weight=None, reduction: str = 'mean', **kwargs):
        super().__init__()
        self.weight = None
        if weight:
            self.weight = torch.tensor(weight)
        self.gamma = gamma

        self.reduction = get_reduction(reduction)

    def __call__(self, x, target, **kwargs):
        ce_loss = F.cross_entropy(x, target, reduction='none', **kwargs)
        pt = torch.exp(-ce_loss)
        if self.weight is not None:
            self.weight = self.weight.to(x.device)
            wtarget = substitute_values(target, self.weight,
                                        unique=torch.arange(len(self.weight), device=target.device))
            focal_loss = torch.pow((1 - pt), self.gamma) * wtarget * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return self.reduction(focal_loss)


def substitute_values(x: torch.Tensor, values, unique=None):
    if unique is None:
        unique = x.unique()
    lt = torch.full((unique.max() + 1,), -1, dtype=values.dtype, device=x.device)
    lt[unique] = values
    return lt[x]


def get_reduction(reduction: str):
    if reduction == 'none':
        return lambda x: x
    elif reduction == 'mean':
        return torch.mean
    elif reduction == 'sum':
        return torch.sum
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
