import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
from utils.cldice import clDice

class AverageMeter(object):
    def __init__(self):
        self.val = []

    def update(self, val):
        self.val.append(val)


    @property
    def mean(self):
        return np.round(np.mean(self.val), 4)

    @property
    def std(self):
        return np.round(np.std(self.val), 4)


def to_one_hot(seg, all_seg_labels=None):
    if all_seg_labels is None:
        all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
        result[i][seg == l] = 1
    return result


def get_metrics(predict, target,run_clDice = False, threshold=0.5):

    
    predict_b = np.where(predict >= threshold, 1, 0)
    
    cldice = clDice(predict_b,target) if run_clDice else 0
    predict = predict.flatten()
    predict_b = predict_b.flatten()
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
    DSC = 2 * pre * sen / (pre + sen)
    return {
        "DSC": np.round(DSC, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "IOU": np.round(iou, 4),
        "AUC": np.round(auc, 4),
        "cldice": np.round(cldice, 4)
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
