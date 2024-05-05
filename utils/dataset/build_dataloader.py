from torch.utils.data import DataLoader


def init_data_loader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def init_data_loaders_weedmapping(dataset, batch_size_train=None, batch_size_val=None, batch_size_test=None):
    dataset.build_data_loaders(train_batch_size=batch_size_train, val_batch_size=batch_size_val, test_batch_size=batch_size_test)
    return dataset.train_loader, dataset.val_loader, dataset.test_loader

