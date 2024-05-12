from torchmetrics import Accuracy, Precision, Recall, F1Score

from utils.metrics.metric_wrapper import MetricWrapper, MockMetric, MODE
from experiments.weed_mapping_experiment.backend.utils import metrics


class MetricsHandler:
    def __init__(self, mode: MODE, num_classes: int, device):
        self.mode = mode
        self.device = device

        if self.mode == 'standard':
            self.TASK = 'multiclass'
            self.NUM_CLASSES = num_classes
            self.AVG = 'macro'
        elif self.mode == 'weed_mapping':
            self.TASK = 'multiclass'
            self.NUM_CLASSES = 3
            self.AVG = 'macro'
        else:
            raise ValueError(f"Mode {mode} not supported.")

        self._init_metrics()

    def _init_metrics(self):
        metric_args = dict(num_classes=self.NUM_CLASSES, task=self.TASK, average=self.AVG)
        full_args = dict(metric_args=metric_args, mode=self.mode, device=self.device)
        if self.mode == 'standard':
            self.accuracy = MetricWrapper(metric=Accuracy, metric_args=dict(num_classes=self.NUM_CLASSES, task=self.TASK), mode=self.mode, device=self.device)
            self.precision = MetricWrapper(metric=Precision, **full_args)
            self.recall = MetricWrapper(metric=Recall, **full_args)
            self.f1 = MetricWrapper(metric=F1Score, **full_args)
        elif self.mode == 'weed_mapping':
            self.accuracy = MockMetric()
            self.precision = MetricWrapper(metric=metrics.Precision, **full_args)
            self.recall = MetricWrapper(metric=metrics.Recall, **full_args)
            self.f1 = MetricWrapper(metric=metrics.F1Score, **full_args)

    def update_metrics(self, pred, y):
        self.accuracy.update(pred, y)
        self.precision.update(pred, y)
        self.recall.update(pred, y)
        self.f1.update(pred, y)

    def compute_metrics(self):
        return self.accuracy.compute(), self.precision.compute(), self.recall.compute(), self.f1.compute()
