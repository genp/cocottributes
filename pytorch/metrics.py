import numpy as np


def average_precision(truth, scores):
    precision, recall = precision_recall(truth, scores)
    ap = voc_ap(recall, precision)
    return ap


def voc_ap(recall, precision):
    """Pascal VOC AP implementation in pure python"""
    mrec = np.hstack((0, recall, 1))
    mpre = np.hstack((0, precision, 0))

    for i in range(mpre.size-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i = np.ravel(np.where(mrec[1:] != mrec[0:-1])) + 1

    ap = np.sum((mrec[i]-mrec[i-1]) * mpre[i])
    return ap


def precision_recall(truth, scores):
    """
    Computer precision-recall curve
    :param truth: the ground truth labels. 1 is positive and -1 is negative label
    :param scores: output confidences from the classifier. Values greater than 0.0 are considered positive detections.
    :return:
    """
    sort_inds = np.argsort(-scores, kind='stable')

    # tp = np.cumsum(truth[sort_inds] == 1)
    # fp = np.cumsum(truth[sort_inds] == -1)
    tp = (truth == 1)[sort_inds]
    fp = (truth == -1)[sort_inds]

    tp = np.cumsum(tp.astype(np.float))
    fp = np.cumsum(fp.astype(np.float))

    npos = (truth == 1).sum()

    recall = tp / npos
    precision = tp / (tp + fp)

    return precision, recall


def hamming_distance(gt, pred):
    """
    Get the percentage of correct positive/negative classifications per attribute.
    :param gt:
    :param pred:
    :return:
    """
    return sum([1 for (g, p) in zip(gt, pred) if g == p]) / float(len(gt))
