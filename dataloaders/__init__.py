import os
from configs import g_conf
from . import carlaImages
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def make_data_loader(model_name, base_dir, train_dataset_names, batch_size, valid_dataset_names, batch_size_eval, rank=0, num_process=1):
    train_set = carlaImages.carlaImages(model_name, base_dir, train_dataset_names, split='train', rank=rank)
    if num_process > 1:
        sampler = DistributedSampler(train_set, num_replicas=num_process, rank=rank, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_set, batch_size=batch_size // num_process, num_workers=g_conf.NUM_WORKER, drop_last=True, shuffle=False, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=g_conf.NUM_WORKER, drop_last=True)

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