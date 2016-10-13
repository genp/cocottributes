coco_root = '/data/hays_lab/COCO/coco/'
import sys
caffe_root = '/home/gen/caffe/'  
sys.path.append(caffe_root+'python')

from sklearn.metrics import average_precision_score, accuracy_score

import caffe
import numpy as np


def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)

# This is checking the baseline if this classifier says Negative to everything
def check_baseline_accuracy(net, num_batches, num_labels, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = np.zeros((batch_size, num_labels))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


def check_ap(net, num_batches, batch_size = 128):
    ap = np.zeros((net.blobs['label'].data.shape[1],1))
    baseline_ap = np.zeros((net.blobs['label'].data.shape[1],1))

    for n in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data
        baseline_ests = -1*np.ones(gts.shape) 
        for dim in range(gts.shape[1]):
            tmp = gts[:,dim]
            fmt_gt = tmp[np.where(tmp!=-1)]
            # fmt_gt[np.where(fmt_gt==0)] = -1
            fmt_est = ests[:,dim]
            fmt_est = fmt_est[np.where(tmp!=-1)]            
            fmt_est_base = baseline_ests[:,dim]
            fmt_est_base = fmt_est_base[np.where(tmp!=-1)]            
            ap_score = average_precision_score(fmt_gt, fmt_est)
            base_ap_score = average_precision_score(fmt_gt, fmt_est_base)
            #print classes[dim] +' ' +str(ap_score)
            ap[dim] = ap_score
            baseline_ap[dim] = base_ap_score

    return ap/float(num_batches), baseline_ap/float(num_batches)
