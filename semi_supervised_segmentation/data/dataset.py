from data.data_augmentation import Compose, ToTensor, CropToFixed, HorizontalFlip, BlobsToMask, VerticalFlip, RandomRotate90, GaussianBlur3D, RandomContrast, AdditiveGaussianNoise, ElasticDeformation, Cutout
from skimage.transform import resize
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
import torch
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import sys
sys.path.append("..")


class label_dataset(Dataset):
    def __init__(self, config, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.size = config.DATASET.PATCH_SIZE
        self.num_each_epoch = config.DATASET.NUM_EACH_EPOCH
        self.images, self.gts = self.read_image(
            self.images_path, self.labels_path)
        self.images = self.images[0:int(config.DATASET.NUM_LABEL)]
        self.gts = self.gts[0:int(config.DATASET.NUM_LABEL)]
        self.config = config
        seed = np.random.randint(123)
        self.seq_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ToTensor(False)
        ])

        self.gt_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ToTensor(False)
        ])

    def read_image(self, images_path, label_path):
        label_files = list(sorted(os.listdir(label_path)))
        images = []
        gts = []
        for i in range(len(label_files)):
            image_each_slice = []
            for j in range(8):
                img = cv2.imread(os.path.join(
                    images_path, f"image_s{i}_i{j}.png"), 0)
                image_each_slice.append(img)
            seq = np.array(image_each_slice)
            mn = seq.mean()
            std = seq.std()
            seq = (seq - mn) / (std + 1e-8)
            images.append(seq)

            gt = cv2.imread(os.path.join(
                label_path, f"label_s{i}.png"), 0)
            gt = np.array(gt/255)[np.newaxis]
            gts.append(gt)

        return images, gts

    def __getitem__(self, idx):

        id = np.random.randint(len(self.images))
        img = self.images[id]
        gt = self.gts[id]

        img = self.seq_DA(img)
        gt = self.gt_DA(gt)
        return img, gt[0].long()

    def __len__(self):
        return self.num_each_epoch


class train_all_dataset(label_dataset):
    def __init__(self, config, images_path, labels_path, unlabel_images_path, pseudo_images_path):

        self.images_path = images_path
        self.labels_path = labels_path
        self.unlabel_images_path = unlabel_images_path
        self.pseudo_images_path = pseudo_images_path
        self.size = config.DATASET.PATCH_SIZE
        self.num_each_epoch = config.DATASET.NUM_EACH_EPOCH
        self.epoch = config.TRAIN.EPOCHS
        self.images, self.gts = self.read_image(
            self.images_path, self.labels_path)
        self.images = self.images[0:int(config.DATASET.NUM_LABEL)]
        self.gts = self.gts[0:int(config.DATASET.NUM_LABEL)]
        self.pl_images, self.pl_gts = self.read_image(
            self.unlabel_images_path, self.pseudo_images_path)
        # self.gts = self.read_label(self.labels_path)
        self.count = 0
        seed = np.random.randint(123)
        self.seq_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ToTensor(False)
        ])

        self.gt_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ToTensor(False)
        ])

        self.strong_seq_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            RandomContrast(np.random.RandomState(
                seed), execution_probability=0.5),
            ElasticDeformation(np.random.RandomState(seed), spline_order=3),
            Cutout(np.random.RandomState(seed)),
            GaussianBlur3D(execution_probability=0.5),
            AdditiveGaussianNoise(np.random.RandomState(
                seed), scale=(0., 0.1), execution_probability=0.1),
            ToTensor(False)
        ])
        self.strong_gt_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ElasticDeformation(np.random.RandomState(seed), spline_order=1),
            BlobsToMask(),
            ToTensor(False)
        ])

    def __getitem__(self, idx):

        if np.random.random() < 0.5:

            id = np.random.randint(len(self.images))
            img = self.images[id]
            gt = self.gts[id]
            img = self.seq_DA(img)
            gt = self.gt_DA(gt)

        else:
            id = np.random.randint(len(self.pl_images)//2)
            img = self.pl_images[id]
            gt = self.pl_gts[id]

            img = self.strong_seq_DA(img)
            gt = self.strong_gt_DA(gt)

        return img, gt[0].long()

    def __len__(self):
        return self.num_each_epoch


class test_dataset(label_dataset):
    def __init__(self, config, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE

        self.images, self.gts = self.read_image(
            self.images_path, self.labels_path)
        self.img_patch = self.get_patch(
            self.images, self.patch_size, self.stride)
        self.gt_patch = self.get_patch(self.gts, self.patch_size, self.stride)

    def get_patch(self, image_list, patch_size, stride):
        patch_list = []
        _, h, w = image_list[0].shape

        pad_h = stride - (h - patch_size[0]) % stride
        pad_w = stride - (w - patch_size[1]) % stride
        for image in image_list:
            image = F.pad(torch.from_numpy(image).float(),
                          (0, pad_w, 0, pad_h), "constant", 0)
            image = image.unfold(1, patch_size[0], stride).unfold(
                2, patch_size[1], stride).permute(1, 2, 0, 3, 4)
            image = image.contiguous().view(
                image.shape[0] * image.shape[1], image.shape[2], patch_size[0], patch_size[1])
            for sub in image:
                patch_list.append(sub)
        return patch_list

    def __getitem__(self, idx):

        img = self.img_patch[idx]
        gt = self.gt_patch[idx]
        return img, gt[0].long()

    def __len__(self):
        return len(self.img_patch)


class inference_dataset(test_dataset):
    def __init__(self, config, images_path):
        self.images_path = images_path
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE

        self.images = self.read_image(self.images_path)
        self.img_patch = self.get_patch(
            self.images, self.patch_size, self.stride)

    def read_image(self, images_path):
        num_files = list(sorted(os.listdir(images_path)))
        images = []
        for i in range(len(num_files)//8):
            image_each_slice = []
            for j in range(8):
                img = cv2.imread(os.path.join(
                    images_path, f"image_s{i}_i{j}.png"), 0)
                image_each_slice.append(img)
            seq = np.array(image_each_slice)
            mn = seq.mean()
            std = seq.std()
            seq = (seq - mn) / (std + 1e-8)
            images.append(seq)

        return images

    def get_patch(self, image_list, patch_size, stride):
        patch_list = []
        _, h, w = image_list[0].shape

        pad_h = stride - (h - patch_size[0]) % stride
        pad_w = stride - (w - patch_size[1]) % stride
        for image in image_list:
            image = F.pad(torch.from_numpy(image).float(),
                          (0, pad_w, 0, pad_h), "constant", 0)
            image = image.unfold(1, patch_size[0], stride).unfold(
                2, patch_size[1], stride).permute(1, 2, 0, 3, 4)
            image = image.contiguous().view(
                image.shape[0] * image.shape[1], image.shape[2], patch_size[0], patch_size[1])
            for sub in image:
                patch_list.append(sub)
        return patch_list

    def __getitem__(self, idx):

        img = self.img_patch[idx]

        return img

    def __len__(self):
        return len(self.img_patch)
