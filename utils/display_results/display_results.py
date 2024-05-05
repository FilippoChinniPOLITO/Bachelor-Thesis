import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.misc.device import get_device


def display_images(images, predictions, real, number_of_images=10):
    num_cols = 10
    num_rows = np.ceil(number_of_images / num_cols).astype(int)

    plt.figure(figsize=(20, 2 * num_rows))
    for i in range(number_of_images):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'[{i}] = {real[i]} : {predictions[i][0]}')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def prediction_loop(dataloader, model):
    y_pred = []
    y_test = []
    images = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(get_device()), target.to(get_device())
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            y_pred.extend(pred.tolist())
            y_test.extend(target.tolist())
            images.extend(data.cpu().numpy())
    return images, y_pred, y_test