#!/usr/bin/env python
import os, sys, pickle, time, math
import argparse

from sklearn import svm
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

from app import db
from app.models import Label, Feature, Patch, Image, Annotation, ClassifierScore
from classifier import Classifier
from mturk import manage_hits
import caffe

def test_db(img_id, is_patch, attr_ids, cat_id, feat_type):
    if is_patch:
        feat = joblib.load(Feature.query.filter(Feature.patch_id == img_id).\
                                         filter(Feature.type == feat_type).\
                                         first().location)
    else:
        feat = joblib.load(Feature.query.filter(Feature.image_id == img_id).\
                                         filter(Feature.type == feat_type).\
                                         first().location)
    return test_feat(feat, attr_ids, cat_id, feat_type)

def test_image(img, attr_ids, cat_id, feat_type, layer_name):
    '''
    img is np.ndarray
    attr_ids are attributes to predict
    feat_type is feature name, corresponding to feature name in db
    layer_name is cnn layer to get activations from
    '''
    # model setup
    MODEL_FILE = '/home/gen/caffe/models/hybridCNN/hybridCNN_deploy_FC7.prototxt'
    # pretrained weights
    PRETRAINED = '/home/gen/caffe/models/hybridCNN/hybridCNN_iter_700000.caffemodel'
    # network setup
    net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load('/home/gen/caffe/models/hybridCNN/hybridCNN_mean.npy').mean(1).mean(1))
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))
    net.blobs['data'].reshape(1,3,227,227)

    ###### TODO: depends on machine
    caffe.set_mode_gpu()
    ######

    net.blobs['data'].data[...] = transformer.preprocess('data',img)
    out = net.forward(blobs=[layer_name])
    feat = out[layer_name]
    
    return test_feat(feat, attr_ids, cat_id, feat_type)
                                         
def test_feat(feat, attr_ids, cat_id, feat_type):                                         
    res = []
    for a in attr_ids:
        c = ClassifierScore.query.\
                            filter(ClassifierScore.type == feat_type).\
                            filter(ClassifierScore.label_id == a).\
                            filter(ClassifierScore.cat_id == cat_id).\
                            first()
        print Label.query.get(c.label_id).name
        print c.id
        mdl = joblib.load(c.location)
        conf = mdl.test(feat)
        res.append(conf)
        
    return sorted(zip(attr_ids, res), key = lambda x: x[1], reverse=True)


def print_result(img, attr_confs, cat_id, sname):

    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')  # clear x- and y-axes
    if cat_id == -1:
        plt.title('all object classifier')
    else:
        plt.title(Label.query.get(cat_id).name)
    for a in attr_confs:
        attr = Label.query.get(a[0]).name
        t = '%s %0.3f' % (attr, a[1])
        plt.text(img.shape[1]+50, ind*100+100, t, ha='left')
    
    fig.savefig(sname, dpi = 300,  bbox_inches='tight')    
    pass    
