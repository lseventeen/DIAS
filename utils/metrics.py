import numpy as np
import cv2
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)


def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


def get_metrics(predict, target, threshold=0.5):

    predict = predict.flatten()
    predict_b = np.where(predict >= threshold, 1, 0)
    target = target.flatten()
    if max(target) > 1:
        target = to_one_hot(target, all_seg_labels=[1]).flatten()
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    if np.all(target == 0) or np.all(predict == 0):
        auc = 1
    else:
        auc = roc_auc_score(target, predict)
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }


def get_color(predict, target):

    tp = (predict * target)
    tn = ((1 - predict) * (1 - target))
    fp = ((1 - target) * predict)
    fn = ((1 - predict) * target)
    H, W = predict.shape
    img_colour = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            if tp[i, j] == 1:
                img_colour[i, j, :] = [255, 255, 255]
            elif tn[i, j] == 1:
                img_colour[i, j, :] = [0, 0, 0]
            elif fp[i, j] == 1:
                img_colour[i, j, :] = [255, 255, 0]
            elif fn[i, j] == 1:
                img_colour[i, j, :] = [114, 128, 250]
    return img_colour


def count_connect_component(predict, target, connectivity=8):

    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n
