import multiprocessing
import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_num_workers():
    return multiprocessing.cpu_count()
