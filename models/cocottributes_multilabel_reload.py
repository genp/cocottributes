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
from sklearn.externals import joblib

import cocottributes_tools as tools #this contains some tools that we need
from cocottributes_multilabel_net import *
from metrics import *
import print_funcs
from app.models import Label

# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)

workdir = './cocottributes_multilabel_with_datalayer'
if not os.path.exists(osp.join(workdir, 'plots')):
    os.makedirs(osp.join(workdir, 'plots'))
# Attribute indices and Patch Instance indices from Cocottributes dataset
obj_attr_supercategory_id = 407
label_ids = [x.id for x in Label.query.filter(Label.parent_id == obj_attr_supercategory_id).order_by(Label.id).all()]
num_labels = len(label_ids)

_train_loss = []
_weight_params = {}    
timestr = time.strftime("%Y%m%d-%H%M%S")
solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
solver.net.copy_from(osp.join(workdir, 'snapshot_iter_101.caffemodel'))
solver.test_nets[0].share_with(solver.net)

def continue_training(num_iter):
    for itt in range(num_iter):
        solver.step(1)
        _train_loss.append(solver.net.blobs['loss'].data) # this should be output from loss layer
        print_funcs.print_layer_params(solver, _weight_params)

        if itt % 1 == 0: # 100 not 1
            print 'itt:{}'.format(itt), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 10))

        print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], len(val_ids)/128, num_labels))
        
        joblib.dump(_weight_params, osp.join(workdir, 'plots/cocottributes_network_parameters_%s.jbl' % timestr), compress=6)
        joblib.dump(_train_loss, osp.join(workdir, 'plots/cocottributes_network_loss_%s.jbl'% timestr), compress=6)



def calc_ap_scores():
    
    baseline_ap = check_baseline_ap(solver.test_nets[0], len(val_ids)/128, num_labels)
    ap = check_ap(solver.test_nets[0], len(val_ids)/128)
    ap_scores = {}
    ap_scores['ap'] = ap
    ap_scores['baseline_ap'] = baseline_ap
    joblib.dump(ap_scores, 'plots/ap_scores.jbl')

def test_ex():    
    image_index = 0 #Lets look at the first image in the batch.
    test_net = solver.test_nets[0]
    test_net.forward()
    #plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
    sname = '/gpfs/main/home/gen/coco_attributes/scratch/test_img.jpg'
    plt.imsave(sname , transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
    gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
    estlist = test_net.blobs['score'].data[image_index, ...] > 0
    print 'Ground truth: ',
    for idx, val in enumerate(gtlist):
        if val:
            print classes[idx] + ',',

    print ''
    print 'Estimated: ',
    for idx, val in enumerate(estlist):
        if val == 1:
            print classes[idx] + ',',                                                     



