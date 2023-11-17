from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from data.dataset import label_dataset, train_all_dataset, test_dataset, inference_dataset
from prefetch_generator import BackgroundGenerator
import torch


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def build_train_single_loader(config):
    train_dataset = label_dataset(
        config, images_path=config.DATASET.TRAIN_IMAGE_PATH, labels_path=config.DATASET.TRAIN_LABEL_PATH)
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
    return train_loader


def build_train_all_loader(config, pseudo_images_path):
    train_dataset = train_all_dataset(config, images_path=config.DATASET.TRAIN_IMAGE_PATH,
                                      labels_path=config.DATASET.TRAIN_LABEL_PATH,
                                      num_unlabel_images = config.DATASET.NUM_UNLABEL,
                                      unlabel_images_path=config.DATASET.UNLABEL_IMAGE_PATH,
                                      pseudo_images_path=pseudo_images_path)
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
    return train_loader


def build_val_loader(config):
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
    return val_loader


def build_test_loader(config):
    dataset = test_dataset(config, images_path=config.DATASET.TEST_IMAGE_PATH,
                           labels_path=config.DATASET.TEST_LABEL_PATH)

    test_loader = DataLoaderX(
        dataset,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )
    return test_loader


def build_inference_loader(config):
    test_dataset = inference_dataset(
        config, images_path=config.DATASET.UNLABEL_IMAGE_PATH)

    test_loader = DataLoaderX(
        test_dataset,
        batch_size=config.DATALOADER.BATCH_SIZE,
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory=config.DATALOADER.PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )
    return test_loader
