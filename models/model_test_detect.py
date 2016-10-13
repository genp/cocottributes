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
from copy import copy
caffe_root = '/home/gen/caffe/'  
sys.path.append(caffe_root+'python')
sys.path.append(cocottributes_root+'caffe/')
sys.path.append(cocottributes_root)

import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.externals import joblib
from pylab import *

from app import db
from app.models import Label, Image, Patch
from models.cocottributes_tools import SimpleTransformer

# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)

# Attribute indices and Patch Instance indices from Cocottributes dataset
obj_attr_supercategory_id = 407
label_ids = [x.id for x in Label.query.filter(Label.parent_id == obj_attr_supercategory_id).order_by(Label.id).all()]
num_labels = len(label_ids)

workdir = '/data/gen_data/COCO/cocottributes_reference_model'
os.chdir(workdir)
num_val_batches = 147#change??
solver = caffe.SGDSolver( 'solver.prototxt')
solver.net.copy_from('snapshot_iter_200000.caffemodel')
solver.test_nets[0].share_with(solver.net)


transformer = caffe.io.Transformer({'data': solver.net.blobs['images'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# setting mean to net's mean image
blob = caffe.proto.caffe_pb2.BlobProto()
data = open("/home/gen/coco_attributes/models/cocottributes_mean.binaryproto", 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
transformer.set_mean('data',arr.reshape((3,227, 227)))

classes = [x.name for x in Label.query.filter(Label.parent_id == obj_attr_supercategory_id).order_by(Label.id).all()]
print classes
good_inds = []
for item in ['angry', 'cuddling', 'eating', 'filling ', 'fresh', 'professional', 'riding']:#, 'sporty']:
    good_inds.append(classes.index(item))
    
def test_image(img, net, classes):
    '''
    img is np.ndarray
    llnet is caffe network
    classes are labels for output dimensions
    '''

    net.blobs['images'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    estlist = net.blobs['score'].data[...]
    estclasses = []
    for idx, val in enumerate(estlist):
        if val:
            estclasses += [classes[idx]]
            
    return estclasses
                                         

def print_result(img, estclasses, gtclasses, sname):
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')  # clear x- and y-axes
    for ind, a in enumerate(estclasses):
        plt.text(img.shape[1]+50, ind*img.shape[0]*0.1, a, ha='left')
    for ind, a in enumerate(gtclasses):
        plt.text(img.shape[1]+200, ind*img.shape[0]*0.1, a, ha='left')
    
    fig.savefig(sname, dpi = 300,  bbox_inches='tight')
    plt.close()

def print_result_bbox(img, x, y, width, height, estclasses, gtclasses, sname):
    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')  # clear x- and y-axes
    for ind, a in enumerate(estclasses):
        plt.text(img.shape[1]+50, ind*img.shape[0]*0.1, a, ha='left')
    for ind, a in enumerate(gtclasses):
        plt.text(img.shape[1]+200, ind*img.shape[0]*0.1, a, ha='left')        
    currentAxis = plt.gca()
    currentAxis.add_patch(patches.Rectangle((x, y), width, height, fill=None, edgecolor="green", alpha=0.5))
    fig.savefig(sname, dpi = 300,  bbox_inches='tight')
    plt.close()
    
#######################################
# Check that things are working
'''
solver.net.forward()

image_index = 0 #Lets look at the first image in the batch
sname = '/gpfs/main/home/gen/coco_attributes/scratch/train_img.jpg'
plt.imsave(sname , transformer.deprocess('data', copy(solver.net.blobs['images'].data[image_index, ...])))
gtlist = solver.net.blobs['labels'].data[image_index, ...].astype(np.int)
est = solver.net.blobs['score'].data[image_index, ...] > 0
#Load classes for printing gt and estimated labels

print 'Num attributes: %d' % len(label_ids)
print 'Ground truth: ',
for idx, val in enumerate(gtlist):
    if val:
        print classes[idx] + ',',
print ''
for idx, val in enumerate(gtlist):
    if val:
        print classes[idx] + ',',
print ''
'''
test_net = solver.test_nets[0]
patch_ids_val = joblib.load('/data/gen_data/COCO/cocottributes_reference_model/patch_ids_val.jbl')
# for batch in range(10):
#     test_net.forward()
for batch in range(0,20):
    # this is moving the set forward by 100 every time...
    test_net.forward()
    for image_index in range(100):
        sname = '/gpfs/main/home/gen/scratch/cocottributes_test_compare/{}.jpg'.format(batch*100+image_index)
        img = transformer.deprocess('data', copy(test_net.blobs['images'].data[image_index, ...]))

        gtlist = test_net.blobs['labels'].data[image_index, ...].astype(np.int)
        estlist = test_net.blobs['score'].data[image_index, ...] > 0

        if not any(estlist[good_inds]):
            continue
        
        print str(image_index) + '-------------------'
        estclasses = []
        estclasses += ['Estimated: ']
        print 'Estimated: ',
        for idx, val in enumerate(estlist):
            if val == 1:
                print classes[idx] + ',',
                estclasses += [classes[idx]]
        print ''
        
        gtclasses = ['Ground truth: ']
        print 'Ground truth: ',
        for idx, val in enumerate(gtlist):
            if val == 1:
                print classes[idx] + ',',
                gtclasses += [classes[idx]]

        print ''

        print_result(img, estclasses, gtclasses, sname)

        patch = Patch.query.get(patch_ids_val[batch*100+image_index])
        img = Image.query.get(patch.image_id)
        sname = '/gpfs/main/home/gen/scratch/cocottributes_test_compare/{}_wholeimg.jpg'.format(batch*100+image_index)
        print_result_bbox(img.to_array(), patch.x, patch.y, patch.width, patch.height, estclasses, gtclasses, sname)
#######################################


