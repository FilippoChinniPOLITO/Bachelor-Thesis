from utils.misc.device import get_device


def train_step(dataloader, model, loss_fn, optimizer, log_step=None):
    # Compute Dateset Size
    size = len(dataloader.dataset) if log_step is not None else None

    # Set the Device
    device = get_device()

    # Set the Model to Training Mode
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Mount data to device
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # (Optional) Print Loss
        if log_step is not None:
            if batch % log_step == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

