import os
from configs import g_conf
from . import carlaImages
from torch.utils.data import DataLoader

def make_data_loader(model_name, base_dir, train_dataset_names, batch_size, valid_dataset_names, batch_size_eval):
    train_set = carlaImages.carlaImages(model_name, base_dir, train_dataset_names, split='train')
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=g_conf.NUM_WORKER,
                              pin_memory=True,  # Is this my Joker moment?
                              drop_last=True)  # we drop the last non-full batch to avoid error
    val_loaders_list=[]
    for valid_dataset_name in valid_dataset_names:
        val_set = carlaImages.carlaImages(model_name, base_dir, [valid_dataset_name], split='val')
        val_loader = DataLoader(val_set,
                                batch_size=batch_size_eval,
                                shuffle=False,
                                num_workers=6,
                                drop_last=True)
        val_loaders_list.append(val_loader)
        return train_loader, val_loaders_list