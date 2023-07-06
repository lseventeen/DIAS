from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from data.dataset import train_dataset, test_dataset, PLC_train_dataset
from prefetch_generator import BackgroundGenerator
import torch


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def build_train_loader(config):
    if config.SCRIBBLE_TYPE == "scribble":
        labels_path = config.DATASET.SCRIBBLE_LABEL_PATH
    elif config.SCRIBBLE_TYPE == "manual":
        labels_path = config.DATASET.MANUAL_LABEL_PATH
    else:
        raise NotImplementedError(f"Unkown param: {config.SCRIBBLE_TYPE}")

    train_dataset = train_dataset(
        config, images_path=config.DATASET.TRAIN_IMAGE_PATH, labels_path=labels_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True) if config.DIS else None
    train_loader = DataLoaderX(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=True if train_sampler is None else False,
        drop_last=True
    )
    val_dataset = test_dataset(
        config, images_path=config.DATASET.VAL_IMAGE_PATH, labels_path=config.DATASET.VAL_LABEL_PATH)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset) if config.DIS else None
    val_loader = DataLoaderX(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=config.DATALOADER.BATCH_SIZE,
        # batch_size=64,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    return train_loader, val_loader


def build_PLC_train_loader(config):
    if config.SCRIBBLE_TYPE == "scribble":
        labels_path = config.DATASET.SCRIBBLE_LABEL_PATH
    elif config.SCRIBBLE_TYPE == "manual":
        labels_path = config.DATASET.MANUAL_LABEL_PATH
    else:
        raise NotImplementedError(f"Unkown param: {config.SCRIBBLE_TYPE}")

    train_dataset = PLC_train_dataset(
        config, images_path=config.DATASET.TRAIN_IMAGE_PATH, labels_path=labels_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True) if config.DIS else None
    train_loader = DataLoaderX(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=True if train_sampler is None else False,
        drop_last=True
    )
    val_dataset = test_dataset(
        config, images_path=config.DATASET.VAL_IMAGE_PATH, labels_path=config.DATASET.VAL_LABEL_PATH)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset) if config.DIS else None
    val_loader = DataLoaderX(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    return train_loader, val_loader


def build_test_loader(config):
    test_dataset = test_dataset(
        config, images_path=config.DATASET.TEST_IMAGE_PATH, labels_path=config.DATASET.TEST_LABEL_PATH)

    test_loader = DataLoaderX(
        test_dataset,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )
    return test_loader
