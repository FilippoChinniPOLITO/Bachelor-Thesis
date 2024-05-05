import logging

import torch
from torch.utils.data import ConcatDataset

from experiments.weed_mapping_experiment.backend.model.base_model import get_param
from experiments.weed_mapping_experiment.backend.dataset.transforms import Denormalize


logger = logging.getLogger(__name__)

default_dataset_params = {"batch_size": 64, "val_batch_size": 200, "test_batch_size": 200, "dataset_dir": "./data/",
                          "s3_link": None}


class DatasetInterface:
    def __init__(self, dataset_params={}, train_loader=None, val_loader=None, test_loader=None, classes=None):

        self.dataset_params = dict(**default_dataset_params)
        self.dataset_params = {**self.dataset_params, **dataset_params}

        self.trainset, self.valset, self.testset = None, None, None
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.classes = classes
        self.batch_size_factor = 1
        self.lib_dataset_params = None

    def build_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None, val_batch_size=None,
                           test_batch_size=None, distributed_sampler: bool = False, pin_memory=True):

        # CHANGE THE BATCH SIZE ACCORDING TO THE NUMBER OF DEVICES - ONLY IN NON-DISTRIBUTED TRAINING MODE
        # IN DISTRIBUTED MODE WE NEED DISTRIBUTED SAMPLERS
        # NO SHUFFLE IN DISTRIBUTED TRAINING

        aug_repeat_count = get_param(self.dataset_params, "aug_repeat_count", 0)
        if aug_repeat_count > 0 and not distributed_sampler:
            raise Exception("repeated augmentation is only supported with DDP.")

        if distributed_sampler:
            None == None
            # self.batch_size_factor = 1
            # train_sampler = RepeatAugSampler(self.trainset,
            #                                  num_repeats=aug_repeat_count) if aug_repeat_count > 0 else DistributedSampler(
            #     self.trainset)
            # val_sampler = DistributedSampler(self.valset)
            # test_sampler = DistributedSampler(self.testset) if self.testset is not None else None
            # train_shuffle = False
        else:
            self.batch_size_factor = batch_size_factor
            train_sampler = None
            val_sampler = None
            test_sampler = None
            train_shuffle = True

        if train_batch_size is None:
            train_batch_size = self.dataset_params['batch_size'] * self.batch_size_factor
        if val_batch_size is None:
            val_batch_size = self.dataset_params['val_batch_size'] * self.batch_size_factor
        if test_batch_size is None:
            test_batch_size = self.dataset_params['test_batch_size'] * self.batch_size_factor

        train_loader_drop_last = get_param(self.dataset_params, 'train_loader_drop_last', default_val=False)

        cutmix = get_param(self.dataset_params, 'cutmix', False)
        cutmix_params = get_param(self.dataset_params, 'cutmix_params')

        # WRAPPING collate_fn
        train_collate_fn = get_param(self.trainset, 'collate_fn')
        val_collate_fn = get_param(self.valset, 'collate_fn')
        test_collate_fn = get_param(self.testset, 'collate_fn')

        if cutmix and train_collate_fn is not None:
            raise Exception("cutmix and collate function cannot be used together")

        self.train_loader = torch.utils.data.DataLoader(self.trainset,
                                                        batch_size=train_batch_size,
                                                        shuffle=train_shuffle,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory,
                                                        sampler=train_sampler,
                                                        collate_fn=train_collate_fn,
                                                        drop_last=train_loader_drop_last)

        self.val_loader = torch.utils.data.DataLoader(self.valset,
                                                      batch_size=val_batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers,
                                                      pin_memory=pin_memory,
                                                      sampler=val_sampler,
                                                      collate_fn=val_collate_fn)

        if self.testset is not None:
            self.test_loader = torch.utils.data.DataLoader(self.testset,
                                                           batch_size=test_batch_size,
                                                           shuffle=False,
                                                           num_workers=num_workers,
                                                           pin_memory=pin_memory,
                                                           sampler=test_sampler,
                                                           collate_fn=test_collate_fn)

        self.classes = self.trainset.classes

    def get_data_loaders(self, **kwargs):

        if self.train_loader is None and self.val_loader is None:
            self.build_data_loaders(**kwargs)

        return self.train_loader, self.val_loader, self.test_loader, self.classes

    def get_val_sample(self, num_samples=1):
        if num_samples > len(self.valset):
            raise Exception("Tried to load more samples than val-set size")
        if num_samples == 1:
            return self.valset[0]
        else:
            return self.valset[0:num_samples]

    def get_dataset_params(self):
        return self.dataset_params

    def print_dataset_details(self):
        logger.info("{} training samples, {} val samples, {} classes".format(len(self.trainset), len(self.valset),
                                                                             len(self.trainset.classes)))

    def undo_preprocess(self, x):
        return (Denormalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std'])(x) * 255).type(torch.uint8)
