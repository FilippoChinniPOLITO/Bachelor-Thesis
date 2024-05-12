import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from experiments.weed_mapping_experiment.backend.utils import metrics
from utils.misc.device import get_device


def eval_step(dataloader, model, loss_fn):
    # Set the Device
    device = get_device()

    # Define Constants for Metrics
    TASK = 'multiclass'
    NUM_CLASSES = 10
    AVG = 'macro'

    # Initialize values for loss
    num_batches = len(dataloader)
    val_loss = 0

    # Initialize metrics
    accuracy = Accuracy(num_classes=NUM_CLASSES, task=TASK).to(device)
    precision = Precision(num_classes=NUM_CLASSES, average=AVG, task=TASK).to(device)
    recall = Recall(num_classes=NUM_CLASSES, average=AVG, task=TASK).to(device)
    f1 = F1Score(num_classes=NUM_CLASSES, average=AVG, task=TASK).to(device)

    # Set the Model to Evaluation Mode
    model.eval()

    # Evaluation Step
    with torch.no_grad():
        for X, y in dataloader:
            # Mount data to device
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss
            pred = model(X)
            val_loss += loss_fn(pred, y).item()

            # Update metrics
            accuracy.update(pred.softmax(dim=1), y)
            precision.update(pred.softmax(dim=1), y)
            recall.update(pred.softmax(dim=1), y)
            f1.update(pred.softmax(dim=1), y)

    avg_loss = val_loss / num_batches

    return avg_loss, accuracy.compute(), precision.compute(), recall.compute(), f1.compute()


def eval_step_weedmapping(dataloader, model, loss_fn):
    # Set the Device
    device = get_device()

    # Define Constants for Metrics
    TASK = 'multiclass'
    NUM_CLASSES = 3
    AVG = 'macro'

    # Initialize values for loss
    num_batches = len(dataloader)
    val_loss = 0

    # Init Evaluation Metrics
    metric_args = dict(num_classes=NUM_CLASSES, task=TASK, average=AVG)
    f1 = metrics.F1(**metric_args).to(device)
    precision = metrics.Precision(**metric_args).to(device)
    recall = metrics.Recall(**metric_args).to(device)

    # Set the Model to Evaluation Mode
    model.eval()

    # Evaluation Step
    with torch.no_grad():
        for X, y in dataloader:
            # Mount data to device
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss
            pred = model(X)
            val_loss += loss_fn(pred, y).item()

            # Update Evaluation Metrics
            f1.update(pred, y)
            precision.update(pred, y)
            recall.update(pred, y)

    avg_loss = val_loss / num_batches

    return avg_loss, f1.compute(), precision.compute(), recall.compute()