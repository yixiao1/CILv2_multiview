import os
from configs import g_conf
from . import carlaImages

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


def make_data_loader(model_name, base_dir, train_dataset_names, batch_size, valid_dataset_names, batch_size_eval, rank=0, num_process=1):
    train_set = carlaImages.carlaImages(model_name, base_dir, train_dataset_names, split='train', rank=rank)
    # Work with a subset of the dataset, if specified by the user in g_conf
    subset_size = int(len(train_set) * max(min(1.0, g_conf.SUBSET_SIZE), 0.0))
    train_set_indices = torch.randperm(len(train_set))[:subset_size]
    train_set = Subset(train_set, train_set_indices)
    if num_process > 1:
        sampler = DistributedSampler(train_set, num_replicas=num_process, rank=rank, shuffle=False, drop_last=True)
        train_loader = DataLoader(train_set, batch_size=batch_size // num_process, num_workers=g_conf.NUM_WORKER,
                                  drop_last=True, shuffle=False, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                  num_workers=g_conf.NUM_WORKER, drop_last=True)

    val_loaders_list=[]
    for valid_dataset_name in valid_dataset_names:
        val_set = carlaImages.carlaImages(model_name, base_dir, [valid_dataset_name], split='val', rank=rank)
        val_loader = DataLoader(val_set,
                                batch_size=batch_size_eval,
                                shuffle=False,
                                num_workers=6,
                                drop_last=True)
        val_loaders_list.append(val_loader)
        return train_loader, val_loaders_list