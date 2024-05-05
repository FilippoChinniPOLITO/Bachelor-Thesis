from torch import Tensor
from torchmetrics import F1Score
from torchmetrics import Precision as TPrecision
from torchmetrics import Recall as TRecall


class F1(F1Score):
    def __init__(self, *args, **kwargs):
        self.component_names = [f"f1_{i}" for i in range(kwargs['num_classes'])] + ['f1']
        super().__init__(**kwargs, average="none", mdmc_average="global")

    def update(self, preds: Tensor, target: Tensor, padding=None) -> None:
        preds = remove_aux(preds)
        if padding is not None:
            for pred, tar in remove_padding(preds, target, padding):
                super().update(pred, tar)
        super().update(preds, target)

    def compute(self):
        per_class_f1 = super().compute()
        return get_multiclass(self.component_names, per_class_f1)


class Precision(TPrecision):
    def __init__(self, *args, **kwargs):
        self.component_names = [f"precision_{i}" for i in range(kwargs['num_classes'])] + ['precision']
        super().__init__(**kwargs, average="none", mdmc_average="global")

    def update(self, preds: Tensor, target: Tensor, padding=None) -> None:
        preds = remove_aux(preds)
        if padding is not None:
            for pred, tar in remove_padding(preds, target, padding):
                super().update(pred, tar)
        super().update(preds, target)

    def compute(self):
        per_class_precision = super().compute()
        return get_multiclass(self.component_names, per_class_precision)


class Recall(TRecall):
    def __init__(self, *args, **kwargs):
        self.component_names = [f"recall_{i}" for i in range(kwargs['num_classes'])] + ['recall']
        super().__init__(**kwargs, average="none", mdmc_average="global")

    def update(self, preds: Tensor, target: Tensor, padding=None) -> None:
        preds = remove_aux(preds)
        if padding is not None:
            for pred, tar in remove_padding(preds, target, padding):
                super().update(pred, tar)
        super().update(preds, target)

    def compute(self):
        per_class_recall = super().compute()
        return get_multiclass(self.component_names, per_class_recall)


def remove_aux(preds):
    if isinstance(preds, tuple):
        return preds[0]
    if isinstance(preds, dict):
        return preds['out']
    # if isinstance(preds, ComposedOutput):
    #     return preds.main
    return preds


def remove_padding(preds, target, padding):
    for i in range(preds.shape[0]):
        w_slice = slice(0, preds.shape[2] - padding[i][1])
        h_slice = slice(0, preds.shape[3] - padding[i][0])
        pred = preds[i, :, w_slice, h_slice]
        targ = target[i, w_slice, h_slice]
        yield pred.unsqueeze(0), targ.unsqueeze(0)


def get_multiclass(names, values):
    macro = values[~values.isnan()].mean()
    per_class = {f"{names[i]}": f1 for i, f1 in enumerate(values)}
    per_class[names[-1]] = macro
    return per_class
