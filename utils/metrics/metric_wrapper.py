from typing import Literal


MODE = Literal['standard', 'weed_mapping']


class MetricWrapper:
    def __init__(self, metric: type, metric_args: dict, mode: MODE, device):
        self.MetricClass = metric
        self.metric_args = metric_args
        self.mode = mode
        self.device = device

        self.metric = self.MetricClass(**self.metric_args).to(self.device)

    def update(self, pred, y):
        if self.mode == 'standard':
            return self.metric.update(pred.softmax(dim=1), y)
        elif self.mode == 'weed_mapping':
            return self.metric.update(pred, y)

    def compute(self):
        return self.metric.compute()


class MockMetric:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute(self):
        return self

    def item(self):
        return self

    def __mul__(self, other):
        return self

    def __format__(self, format_spec):
        return 'N/A  '

    def __round__(self, n=None):
        return self

    def __str__(self):
        return 'N/A  '
