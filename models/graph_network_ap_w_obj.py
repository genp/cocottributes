#!/usr/bin/env python

###
# Pull wieghts out of model snapshots and graph
# 
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

from app import db
from app.models import Label
from mturk import manage_hits
import cocottributes_tools as tools #this contains some tools that we need
import print_funcs

# Note: will be using caffenet reference model weights
#stmt = "select * from (select patch_id from (select a.patch_id, count(distinct label_id) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count > 175) as foo intersect select p.id from patch p, image im where p.image_id = im.id and im.type = 'val2014'"
stmt = "select patch_id from (select a.patch_id, count(distinct label_id) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count > 175"
val_ids = [x[0] for x in db.engine.execute(stmt).fetchall()]
print 'Val set size: {}'.format(len(val_ids))

# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)

#workdir = '/data/hays_lab/COCO/caffemodels_exhaustive_val/finetune_model_m10_w5e-3/'
workdir = '/data/gen_data/COCO/caffemodels_exhaustive_w_obj/finetune_model_adagrad/'
os.chdir(workdir)
if not osp.exists(workdir):
    os.makedirs(workdir)
if not osp.exists(osp.join(workdir, 'plots')):
    os.makedirs(osp.join(workdir, 'plots'))

# Attribute indices and Patch Instance indices from Cocottributes dataset
#Load classes for printing gt and estimated labels
full_obj_lbls = Label.query.filter(Label.parent_id.in_([1,91,93,97])).order_by(Label.id).all()
obj_lbls = [x.id for x in full_obj_lbls]
parent_lbls = [x.parent_id for x in full_obj_lbls]
obj_attr_supercategory_id = 407
label_ids = [x.id for x in Label.query.filter(Label.parent_id == obj_attr_supercategory_id).order_by(Label.id).all()]+sorted(set(obj_lbls + parent_lbls))
num_labels = len(label_ids)
        
# Objects for logging solver training

solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
#itt = 200000
itt=129000
batch_size = 100
solver.net.copy_from(osp.join(workdir, 'snapshot_iter_{}.caffemodel'.format(itt)))
solver.test_nets[0].share_with(solver.net)

from cocottributes_tools import SimpleTransformer
from copy import copy
transformer = SimpleTransformer() # this is simply to add back the bias, re-shuffle the color channels to RGB, and so on...

image_index = 0 #Lets look at the first image in the batch.
#plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
sname = '/gpfs/main/home/gen/coco_attributes/scratch/train_img.jpg'
plt.imsave(sname , transformer.deprocess(copy(solver.net.blobs['images'].data[image_index, ...])))
gtlist = solver.net.blobs['labels'].data[image_index, ...].astype(np.int)



classes = [Label.query.get(x).name for x in label_ids]
print 'Num attributes: %d' % len(classes)
print 'Ground truth: ',
for idx, val in enumerate(gtlist):
    if val:
        print classes[idx] + ',',
print ''



def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = batch_size):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['labels'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)
# This is checking the baseline if this classifier says Negative to everything        
def check_baseline_accuracy(net, num_batches, num_labels, batch_size = batch_size):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['labels'].data
        ests = np.zeros((batch_size, num_labels))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)




from sklearn.metrics import average_precision_score, accuracy_score
def check_baseline_ap(net, num_batches, batch_size = batch_size):
    ap = np.zeros((len(classes),1))
    counts = np.zeros((len(classes),1))
    for n in range(num_batches):
        print '{0} / {1}'.format(n, num_batches)
        net.forward()
        gts = net.blobs['labels'].data
        ests = -1*np.ones(gts.shape)
        for dim in range(gts.shape[1]):
            tmp = gts[:,dim].reshape(gts.shape[0],1)
            fmt_gt = tmp[np.where(tmp!=-1)]
            fmt_gt[np.where(fmt_gt==0)] = -1
            fmt_est = ests[:,dim].reshape(ests.shape[0],1)            
            fmt_est = fmt_est[np.where(tmp!=-1)]
            ap_score = average_precision_score(fmt_gt, fmt_est)
            if not np.isnan(ap_score):
                ap[dim] += ap_score
                counts[dim] += 1

    return ap/counts


def check_ap(net, num_batches, batch_size = batch_size):
    ap = np.zeros((len(classes),1))    
    counts = np.zeros((len(classes),1))
    for n in range(num_batches):
        print '{0} / {1}'.format(n, num_batches)
        net.forward()
        gts = net.blobs['labels'].data
        ests = net.blobs['score'].data
        for dim in range(gts.shape[1]):
            tmp = gts[:,dim].reshape(gts.shape[0],1)
            fmt_gt = tmp[np.where(tmp!=-1)]
            fmt_gt[np.where(fmt_gt==0)] = -1
            fmt_est = ests[:,dim].reshape(ests.shape[0],1)            
            fmt_est = fmt_est[np.where(tmp!=-1)]
            ap_score = average_precision_score(fmt_gt, fmt_est)
            if not np.isnan(ap_score):
                ap[dim] += ap_score
                counts[dim] += 1

    return ap/counts


ap = check_ap(solver.test_nets[0], len(val_ids)/batch_size)
#baseline_ap = check_baseline_ap(solver.test_nets[0], len(val_ids)/batch_size)
ap_scores = {}
ap_scores['label_ids'] = label_ids
ap_scores['ap'] = ap
#ap_scores['baseline_ap'] = baseline_ap
joblib.dump(ap_scores, osp.join(workdir, 'plots/ap_scores_%d.jbl' % itt), compress=6)
    
print 'itt:{}'.format(itt), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 20))
print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 20, num_labels))

cur_ind = 0
    
for image_index in range(cur_ind, cur_ind+100):
    test_net = solver.test_nets[0]
    test_net.forward()
   
    gtlist = test_net.blobs['labels'].data[image_index % batch_size, ...].astype(np.int)
    estlist = test_net.blobs['score'].data[image_index % batch_size, ...] > 0
    print 'Image {} ********'.format(image_index)
    gtstr = 'Ground truth: \n'
    for idx, val in enumerate(gtlist):
        if val == 1:
           gtstr += classes[idx] + ', \n'

    eststr = 'Estimated: \n'
    for idx, val in enumerate(estlist):
        if val == 1:
            eststr += classes[idx] + ', \n'                                                                
    labelstr = ''
    labelstr += 'hamming dist: {}\n'.format(scipy.spatial.distance.hamming(gtlist, estlist)*len(label_ids))
    labelstr += 'baseline hamming dist: {}\n'.format(scipy.spatial.distance.hamming(gtlist, np.zeros((len(gtlist),1)))*len(label_ids))
    fig= plt.figure(figsize=(7, 4))
    plt.title('Test Image {} ********'.format(image_index))
    plt.imshow(transformer.deprocess(copy(test_net.blobs['images'].data[image_index % batch_size, ...])))
    plt.text(-50, 0, gtstr, horizontalalignment='right',
            verticalalignment='top',
            size=10)
    plt.text(230, 0, eststr, horizontalalignment='left',
            verticalalignment='top',
            size=10) 
    sname = osp.join(workdir, 'figs', 'test_img_{}_{}.jpg'.format(itt, image_index))
    plt.savefig(sname)
    print labelstr
    print gtstr
    print eststr
    cur_ind += 1
