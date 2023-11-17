import os
import pickle
import random
import numpy as np
import torch
import cv2
import numpy as np
import torch
import random
import os
from batchgenerators.utilities.file_and_folder_operations import *


def load_checkpoint(checkpoint_path, is_best=False):
    checkpoint_file = "best_checkpoint.pth" if is_best else "final_checkpoint.pth"
    checkpoint = torch.load(
        join(checkpoint_path, checkpoint_file), map_location=torch.device('cpu'))
    return checkpoint


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_cuda(data, non_blocking=True):
    if isinstance(data, list):
        data = [i.cuda(non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(non_blocking=non_blocking)
    return data


def load_checkpoint(checkpoint_path, is_best=True):
    checkpoint_file = "best_model.pth" if is_best else "final_checkpoint.pth"
    checkpoint = torch.load(
        join(checkpoint_path, checkpoint_file), map_location=torch.device('cpu'))
    return checkpoint


def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def read_pickle(path, type):
    with open(file=path + f"/{type}.pkl", mode='rb') as file:
        img = pickle.load(file)
    return img


def save_pickle(path, type, img_list):
    with open(file=path + f"/{type}.pkl", mode='wb') as file:
        pickle.dump(img_list, file)


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape) == 4)
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img
    full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    k = 0
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[
                    k]
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h,
                         w * stride_w:(w * stride_w) + patch_w] += 1
                k += 1
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)
    final_avg = full_prob / full_sum
    return final_avg


def double_threshold_iteration(index, img, h_thresh, l_thresh, save=True):
    h, w = img.shape
    img = np.array(torch.sigmoid(img).cpu().detach()*255, dtype=np.uint8)
    bin = np.where(img >= h_thresh*255, 255, 0).astype(np.uint8)
    gbin = bin.copy()
    gbin_pre = gbin-1
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):
            for j in range(w):
                if gbin[i][j] == 0 and img[i][j] < h_thresh*255 and img[i][j] >= l_thresh*255:
                    if gbin[i-1][j-1] or gbin[i-1][j] or gbin[i-1][j+1] or gbin[i][j-1] or gbin[i][j+1] or gbin[i+1][j-1] or gbin[i+1][j] or gbin[i+1][j+1]:
                        gbin[i][j] = 255

    if save:
        cv2.imwrite(f"save_picture/bin{index}.png", bin)
        cv2.imwrite(f"save_picture/gbin{index}.png", gbin)
    return gbin/255