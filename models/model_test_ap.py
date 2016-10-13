#!/usr/bin/env python

###
# Multilabel classification for MS COCO Attributes
# This network developed from framework introduced by Oscar Beijborn in https://github.com/BVLC/caffe/pull/3471
###
coco_root = '/data/hays_lab/COCO/coco/'
cocottributes_root = '/home/gen/coco_attributes/'
# import some modules
import sys, os, time
import os.path as osp
caffe_root = '/home/gen/caffe/'  
sys.path.append(caffe_root+'python')
sys.path.append(cocottributes_root+'caffe/')
sys.path.append(cocottributes_root)

import caffe
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.externals import joblib
from pylab import *

from app.models import Label
from mturk import manage_hits
import models.cocottributes_tools as tools #this contains some tools that we need

# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)

# Attribute indices and Patch Instance indices from Cocottributes dataset
obj_attr_supercategory_id = 407
label_ids = [x.id for x in Label.query.filter(Label.parent_id == obj_attr_supercategory_id).order_by(Label.id).all()]
num_labels = len(label_ids)

workdir = '/data/gen_data/COCO/cocottributes_reference_model'
os.chdir(workdir)
num_val_batches = 147
solver = caffe.SGDSolver( 'solver.prototxt')
solver.net.copy_from('snapshot_iter_200000.caffemodel')

solver.test_nets[0].share_with(solver.net)


from models.cocottributes_tools import SimpleTransformer
from copy import copy
transformer = SimpleTransformer() # this is simply to add back the bias, re-shuffle the color channels to RGB, and so on...

image_index = 0 #Lets look at the first image in the batch
sname = '/gpfs/main/home/gen/coco_attributes/scratch/train_img.jpg'
plt.imsave(sname , transformer.deprocess(copy(solver.net.blobs['images'].data[image_index, ...])))
gtlist = solver.net.blobs['labels'].data[image_index, ...].astype(np.int)

#Load classes for printing gt and estimated labels
from app.models import Label
classes = [x.name for x in Label.query.filter(Label.parent_id == obj_attr_supercategory_id).order_by(Label.id).all()]
print 'Num attributes: %d' % len(label_ids)
print 'Ground truth: ',
for idx, val in enumerate(gtlist):
    if val:
        print classes[idx] + ',',
print ''



def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['labels'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)
# This is checking the baseline if this classifier says Negative to everything        
def check_baseline_accuracy(net, num_batches, num_labels, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['labels'].data
        ests = np.zeros((batch_size, num_labels))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)

from sklearn.metrics import average_precision_score, accuracy_score
def check_ap(net, num_batches, batch_size = 128):
    ap = np.zeros((net.blobs['labels'].data.shape[1],1))
    baseline_ap = np.zeros((net.blobs['labels'].data.shape[1],1))
    for n in range(num_batches):
        net.forward()
        gts = net.blobs['labels'].data.reshape(net.blobs['labels'].data.shape[:2])
        ests = net.blobs['score'].data.reshape(net.blobs['labels'].data.shape[:2])
        baseline_ests = np.zeros(gts.shape) 
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


ap, baseline_ap = check_ap(solver.test_nets[0], num_val_batches)
ap_scores = {}
ap_scores['ap'] = ap
ap_scores['baseline_ap'] = baseline_ap
print '*** Mean AP and Baseline AP scores {}***'.format(0)
print np.mean([a if not np.isnan(a) else 0 for a in ap])
print np.mean([a if not np.isnan(a) else 0 for a in baseline_ap])

joblib.dump(ap_scores, osp.join(workdir, 'ap_scores.jbl'), compress=6)

    
print 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], num_val_batches))
print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], num_val_batches, num_labels))

image_index = 0 #Lets look at the first image in the batch.
test_net = solver.test_nets[0]
test_net.forward()
#plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
sname = '/gpfs/main/home/gen/coco_attributes/scratch/test_img.jpg'
plt.imsave(sname , transformer.deprocess(copy(solver.net.blobs['images'].data[image_index, ...])))
gtlist = test_net.blobs['labels'].data[image_index, ...].astype(np.int)
estlist = test_net.blobs['score'].data[image_index, ...] > 0
print 'Ground truth: ',
for idx, val in enumerate(gtlist):
    if val == 1:
        print classes[idx] + ',',

print ''
print 'Estimated: ',
for idx, val in enumerate(estlist):
    if val == 1:
        print classes[idx] + ',',                                                                
