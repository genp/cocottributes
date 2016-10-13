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
import cocottributes_tools as tools #this contains some tools that we need
import print_funcs

# Note: will be using caffenet reference model weights

# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# main netspec wrapper
def caffenet_multilabel(data_layer_params, datalayer, num_labels):
    # setup the python data layer
    n = caffe.NetSpec()
    # Python datalayer with two top input layers - one for labels and one for images
    n.data, n.label = L.Python(module = 'cocottributes_multilabel_datalayers', layer = datalayer,
                               ntop = 2, param_str=str(data_layer_params))

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=num_labels)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    
    return str(n.to_proto())

workdir = './cocottributes_multilabel_with_datalayer_finetune00001'
if not osp.exists(workdir):
    os.makedirs(workdir)
if not osp.exists(osp.join(workdir, 'plots')):
    os.makedirs(osp.join(workdir, 'plots'))
solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
solverprototxt.sp['display'] = "1"
solverprototxt.sp['base_lr'] = "0.00001"
solverprototxt.sp['snapshot'] = "50"
solverprototxt.sp['snapshot_prefix'] = '"'+osp.join(workdir, 'snapshot')+'"'
solverprototxt.sp['solver_mode'] = 'GPU'
solverprototxt.write(osp.join(workdir, 'solver.prototxt'))

# Attribute indices and Patch Instance indices from Cocottributes dataset
obj_attr_supercategory_id = 407
label_ids = [x.id for x in Label.query.filter(Label.parent_id == obj_attr_supercategory_id).order_by(Label.id).all()]
num_labels = len(label_ids)

# write train and val nets.
with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
    # provide parameters to the data layer as a python dictionary. Easy as pie!
    # TODO change imsize to 128 - correction for 11x11 conv1 filters...
    data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'train2014', label_ids = label_ids, exhaustivelbls = True)
    f.write(caffenet_multilabel(data_layer_params, 'CocottributesMultilabelDataLayerSync', num_labels))

with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
    data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'val2014', label_ids = label_ids, exhaustivelbls = True)
    f.write(caffenet_multilabel(data_layer_params, 'CocottributesMultilabelDataLayerSync', num_labels))

# Objects for logging solver training
_train_loss = []
_weight_params = {}    

solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
# solver.net.copy_from(osp.join(workdir, 'snapshot_iter_25.caffemodel'))
# solver.restore(osp.join(workdir, 'snapshot_iter_25.solverstate'))
solver.test_nets[0].share_with(solver.net)
t0 = time.clock()
solver.step(1)
print time.clock() - t0, "seconds process time"


_train_loss.append(solver.net.blobs['loss'].data) # this should be output from loss layer
print_funcs.print_layer_params(solver, _weight_params)
timestr = time.strftime("%Y%m%d-%H%M%S")
joblib.dump(_weight_params, osp.join(workdir, 'plots/cocottributes_network_parameters_%s.jbl' % timestr), compress=6)

from cocottributes_tools import SimpleTransformer
from copy import copy
transformer = SimpleTransformer() # this is simply to add back the bias, re-shuffle the color channels to RGB, and so on...

image_index = 0 #Lets look at the first image in the batch.
#plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
sname = '/gpfs/main/home/gen/coco_attributes/scratch/train_img.jpg'
plt.imsave(sname , transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)

#Load classes for printing gt and estimated labels
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




from sklearn.metrics import average_precision_score, accuracy_score
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


ap, baseline_ap = check_ap(solver.test_nets[0], 5)
ap_scores = {}
ap_scores['ap'] = ap
ap_scores['baseline_ap'] = baseline_ap
print '*** Mean AP and Baseline AP scores {}***'.format(0)
print np.mean([a if not np.isnan(a) else 0 for a in ap])
print np.mean([a if not np.isnan(a) else 0 for a in baseline_ap])

for itt in range(100):#500):
    solver.step(1)
    _train_loss.append(solver.net.blobs['loss'].data) # this should be output from loss layer
    print_funcs.print_layer_params(solver, _weight_params)

    if itt % 10 == 0: # 100 not 1
        ap, baseline_ap = check_ap(solver.test_nets[0], 5)
        ap_scores = {}
        ap_scores['ap'] = ap
        ap_scores['baseline_ap'] = baseline_ap
        print '*** Mean AP and Baseline AP scores {}***'.format(itt)
        print np.mean(ap)
        print np.mean(baseline_ap)
        joblib.dump(ap_scores, osp.join(workdir, 'plots/ap_scores_%d.jbl' % itt), compress=6)
        
    joblib.dump(_weight_params, osp.join(workdir, 'plots/cocottributes_network_parameters_%s.jbl' % timestr), compress=6)
    joblib.dump(_train_loss, osp.join(workdir, 'plots/cocottributes_network_loss_%s.jbl'% timestr), compress=6)

baseline_ap = check_baseline_ap(solver.test_nets[0], len(val_ids)/128, num_labels)
ap = check_ap(solver.test_nets[0], len(val_ids)/128)
ap_scores = {}
ap_scores['ap'] = ap
ap_scores['baseline_ap'] = baseline_ap
print '*** Mean AP and Baseline AP scores {}***'.format(itt)
print np.mean([a if not np.isnan(a) else 0 for a in ap])
print np.mean([a if not np.isnan(a) else 0 for a in baseline_ap])
joblib.dump(ap_scores, osp.join(workdir, 'plots/ap_scores_%d.jbl' % itt))
    
print 'itt:{}'.format(itt), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 2))
print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], len(val_ids)/128, num_labels))

image_index = 0 #Lets look at the first image in the batch.
# test_net = solver_async.test_nets[0]
test_net = solver.test_nets[0]
test_net.forward()
#plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
sname = '/gpfs/main/home/gen/coco_attributes/scratch/test_img.jpg'
plt.imsave(sname , transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
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
