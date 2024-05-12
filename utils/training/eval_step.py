import torch

from experiments.weed_mapping_experiment.backend.model.lawin_model import Lawin
from utils.misc.device import get_device
from utils.metrics.metrics_handler import MetricsHandler


def eval_step(dataloader, model, loss_fn):
    # Set the Device
    device = get_device()

    # Initialize values for loss
    num_batches = len(dataloader)
    val_loss = 0

    # Initialize metrics
    mode = get_metrics_mode(model)
    metrics_handler = MetricsHandler(mode=mode, num_classes=10, device=device)

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
            metrics_handler.update_metrics(pred, y)

    avg_loss = val_loss / num_batches

    return avg_loss, metrics_handler.compute_metrics()


def get_metrics_mode(model):
    if isinstance(model, Lawin):
        return 'weed_mapping'
    return 'standard'

