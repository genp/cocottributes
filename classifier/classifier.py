#!/usr/bin/env python
import os, time, sys
import argparse

from random import shuffle
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from app import db
from app.models import Label, Feature, Patch, Image, Annotation, ClassifierScore, AnnotationVecMatView
from mturk import manage_hits

"""

Object for classifier classifier. 
Contains methods for creating, updating, and applying
classifier.

"""

class Classifier:
    def __init__(self):
        self.mdl = svm.LinearSVC()
        self.mdl.set_params(dual=True)
        self.mdl.set_params(C=1.0)
        self.mdl.set_params(verbose=True)
        self.mdl.set_params(class_weight='auto')

    def make_nonlinear(self):
        self.mdl = None
        self.mdl = svm.SVC()

    def get_params(self):
        return self.mdl.get_params()
    
    def train(self, train_set, train_lbls):
        tic = time.time()
        self.mdl.fit(train_set, train_lbls)
        print self.get_params() 
        print 'Training score: '+str(self.mdl.score(train_set, train_lbls))
        toc = time.time()-tic
        print 'Time elapsed: '+str(toc)

    def test(self, test_set):
        conf = self.mdl.decision_function(test_set)
        return conf

    def save(self, save_name):
        joblib.dump(self, save_name, compress=6)


# method for collecting features for given label
def get_examples(label_id, cat_id, feat_type, train_percent, use_whole_img=False):
    pos_patches = {}
    neg_patches = {}
    if cat_id == -1: # case for whole images, use all categories, etc.
        pos_patches['train'] = manage_hits.find_positives([label_id], [], [], 'train2014')
        neg_patches['train'] = manage_hits.find_positives([], [label_id], [], 'train2014')
        pos_patches['val'] = manage_hits.find_positives([label_id], [], [], 'val2014')
        neg_patches['val'] = manage_hits.find_positives([], [label_id], [], 'val2014')
    else:

        cat = Label.query.get(cat_id)
        if cat.parent_id == None:
            cat_ids = [x.id for x in Label.query.filter(Label.parent_id == cat_id).all()]
            print cat_ids
            pos_patches['train'] = []
            neg_patches['train'] = []
            pos_patches['val'] = []
            neg_patches['val'] = []
            for c in cat_ids:
                pos_patches['train'] += manage_hits.find_positives([label_id], [], c, 'train2014')
                neg_patches['train'] += manage_hits.find_positives([], [label_id], c, 'train2014')
                pos_patches['val'] += manage_hits.find_positives([label_id], [], c, 'val2014')
                neg_patches['val'] += manage_hits.find_positives([], [label_id], c, 'val2014')                
        else:
            pos_patches['train'] = manage_hits.find_positives([label_id], [], cat_id, 'train2014')
            neg_patches['train'] = manage_hits.find_positives([], [label_id], cat_id, 'train2014')
            pos_patches['val'] = manage_hits.find_positives([label_id], [], cat_id, 'val2014')
            neg_patches['val'] = manage_hits.find_positives([], [label_id], cat_id, 'val2014')            

    inter = set(pos_patches['train']).intersection(neg_patches['train'])
    for item in inter:
        if label_id > 407:
            pos_patches['train'].remove(item)
        neg_patches['train'].remove(item)

    inter = set(pos_patches['val']).intersection(neg_patches['val'])
    for item in inter:
        if label_id > 407:
            pos_patches['val'].remove(item)
        neg_patches['val'].remove(item)


    # if train percent == -1 or -2, remove items that are in the partial or exhaustive set respectively
    if train_percent < 0 and train_percent > -4:
        stmt = "select patch_id from (select a.patch_id, count(distinct label_id) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count < 175"
        part_patch_ids = [x[0] for x in db.engine.execute(stmt).fetchall()]

        # remove partially labeled items from validation set
        for item in part_patch_ids:
            if item in pos_patches['val']:
                pos_patches['val'].remove(item)
            if item in neg_patches['val']:
                neg_patches['val'].remove(item)

        # if only want to train on exhaustively labeled items, remove partial labels
        if train_percent == -1:
            for item in part_patch_ids:
                if item in pos_patches['train']:
                    pos_patches['train'].remove(item)
                if item in neg_patches['train']:                    
                    neg_patches['train'].remove(item)

        # if only want to train on partial labeled items, remove exhaustive labels
        elif train_percent == -2 or train_percent == -3 or train_percent == -4:
            # TODO: This is actually the wrong way to get the exhaustive set. Update.
            stmt = "select patch_id from (select a.patch_id, count(distinct label_id) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count >= 175"
            ex_patch_ids = [x[0] for x in db.engine.execute(stmt).fetchall()]
            for item in ex_patch_ids:
                if item in pos_patches['train']:
                    pos_patches['train'].remove(item)
                if item in neg_patches['train']:                    
                    neg_patches['train'].remove(item)
            # adjust training set size to be same as in exhaustive classifiers
            if train_percent == -3:
                match_classifier = ClassifierScore.query.filter(ClassifierScore.train_percent == -1).filter(ClassifierScore.label_id == label_id).first()
                pos_patches['train'] = pos_patches['train'][:min(match_classifier.num_pos, len(pos_patches['train']))]
                neg_patches['train'] = neg_patches['train'][:min(match_classifier.num_neg, len(neg_patches['train']))]
    # use the same training and test set as the CNN
    elif train_percent == -4:
        cnn_train_ids = set(joblib.load('/data/gen_data/COCO/cocottributes_reference_model/patch_ids_train.jbl'))
        cnn_val_ids = set(joblib.load('/data/gen_data/COCO/cocottributes_reference_model/patch_ids_val.jbl'))
        pos_patches['train'] = list(set(pos_patches['train']).intersection(cnn_train_ids))
        neg_patches['train'] = list( cnn_train_ids - set(pos_patches['train']))

        pos_patches['val'] = list(set(pos_patches['val']).intersection(cnn_val_ids))
        neg_patches['val'] = list( cnn_val_ids - set(pos_patches['val']))

    # use the same training and test set as the CNN except only KNOWN negatives, not assumed negatives
    elif train_percent == -5:
        cnn_train_ids = set(joblib.load('/data/gen_data/COCO/cocottributes_reference_model/patch_ids_train.jbl'))
        cnn_val_ids = set(joblib.load('/data/gen_data/COCO/cocottributes_reference_model/patch_ids_val.jbl'))
        pos_patches['train'] = list(set(pos_patches['train']).intersection(cnn_train_ids))
        neg_patches['train'] = list(set(neg_patches['train']).intersection(cnn_train_ids))

        pos_patches['val'] = list(set(pos_patches['val']).intersection(cnn_val_ids))
        neg_patches['val'] = list(set(neg_patches['val']).intersection(cnn_val_ids))
        
                                        
    print 'Num pos patches {}'.format(len(pos_patches['train']))
    print 'Num neg patches {}'.format(len(neg_patches['train']))
    if len(pos_patches['train']) == 0 or len(neg_patches['train']) == 0:
        print 'either no positive or no negative training examples'
        return {}, len(pos_patches['train']), len(neg_patches['train']), []
    # Balance train set so same ratio of pos/neg as val set
    if train_percent != -4:
        ratio_train = float(len(pos_patches['train']))/len(neg_patches['train'])
        ratio_val = float(len(pos_patches['val']))/len(neg_patches['val'])
        print 'Trainig ratio: {}'.format(ratio_train)
        print 'Val ratio: {}'.format(ratio_val)
        eps = 0.002
        if ratio_train < ratio_val-eps:
            new_num_neg = int(len(neg_patches['train'])*float(len(neg_patches['val'])*len(pos_patches['train']))/(len(pos_patches['val'])*len(neg_patches['train'])))
            shuffle(neg_patches['train'])
            neg_patches['train'] = neg_patches['train'][:new_num_neg]
        elif ratio_train > ratio_val+eps:
            new_num_pos = int(len(pos_patches['train'])*float(len(pos_patches['val'])*len(neg_patches['train']))/(len(neg_patches['val'])*len(pos_patches['train'])))
            shuffle(pos_patches['train'])
            pos_patches['train'] = pos_patches['train'][:new_num_pos]
            
        print 'Rebalanced datasets:'
        print 'Num pos patches {}'.format(len(pos_patches['train']))
        print 'Num neg patches {}'.format(len(neg_patches['train']))

    for tp in ['train', 'val']:
        pos_feat = []
        neg_feat = []
        missing_p = []
        for idx, p in enumerate(pos_patches[tp]):
            try:
                if use_whole_img:
                    img_id = Patch.query.get(p).image.id
                    feat = joblib.load(Feature.query.\
                                              filter(Feature.image_id == img_id).\
                                              filter(Feature.type == feat_type).\
                                              first().location)
                    # print '%d img_id %d' % (idx, img_id)
                else: 
                    # print '%d patch_id %d' % (idx, p)
                    if feat_type == 'attributes':
                        feat = np.array(get_multilabel(p))
                        # print 'loaded feat {}/{} {}'.format(idx, len(pos_patches[tp]), tp) 
                    else:
                        feat = joblib.load(Feature.query.\
                                                  filter(Feature.patch_id == p).\
                                                  filter(Feature.type == feat_type).\
                                                  first().location)                
                if pos_feat == []:
                    pos_feat = feat
                else:
                    pos_feat = np.vstack([pos_feat, feat])                                      
            except AttributeError, e:
                missing_p.append(p)
                
        for idx, p in enumerate(neg_patches[tp]):
            try:
                if use_whole_img:
                    img_id = Patch.query.get(p).image.id
                    feat = joblib.load(Feature.query.\
                                              filter(Feature.image_id == img_id).\
                                              filter(Feature.type == feat_type).\
                                              first().location)
                    # print '%d img_id %d' % (idx, img_id)              
                    
                else:
                    # print '%d patch_id %d' % (idx, p)
                    if feat_type == 'attributes':
                        feat = np.array(get_multilabel(p))                        
                        # print 'loaded feat {}/{} {}'.format(idx, len(neg_patches[tp]), tp)                                              
                    else:                    
                        feat = joblib.load(Feature.query.\
                                                  filter(Feature.patch_id == p).\
                                                  filter(Feature.type == feat_type).\
                                                  first().location)                
                if neg_feat == []:
                    neg_feat = feat
                else:
                    neg_feat = np.vstack([neg_feat, feat])                                                                            
            except AttributeError, e:
                missing_p.append(p)

        if len(pos_feat) == 0 or len(neg_feat) == 0:
            print 'either no positive or no negative training examples'
            return {}, len(pos_feat), len(neg_feat), []
            
        if tp == 'train':        
            num_pos_train = pos_feat.shape[0]
            num_neg_train = neg_feat.shape[0]
            train = np.vstack([pos_feat, neg_feat])
            train_lbls = np.vstack([np.ones((num_pos_train,1)), -1*np.ones((num_neg_train,1))])
        else:
            num_pos_test = pos_feat.shape[0]
            num_neg_test = neg_feat.shape[0]
            test = np.vstack([pos_feat, neg_feat])        
            test_lbls = np.vstack([np.ones((num_pos_test,1)), -1*np.ones((num_neg_test,1))])    
                    
    examples = {}
    examples['train'] = train
    examples['train_lbls'] = train_lbls
    examples['test'] = test
    examples['test_lbls'] = test_lbls
    return examples, num_pos_train, num_neg_train, missing_p

# train classifier for given label        
def make_classifier(label_id, cat_id, feat_type, train_percent, spath, use_whole_img=False, weighted_ap=False):
    cat = Label.query.get(cat_id)
    if cat == None:
        cat = 'all'
    else:
        cat = cat.name
    lbl = Label.query.get(label_id).name    
    sname = os.path.join(spath, feat_type, '{}_{}_{}.jbl'.format(cat,slugify(lbl),train_percent))
    if os.path.exists(sname):
        print "The model at {} already exists :)".format(sname)
        cls = joblib.load(sname)
        examples, num_pos, num_neg, mp = get_examples(label_id, cat_id, feat_type, train_percent, use_whole_img)
    else:
        examples, num_pos, num_neg, mp = get_examples(label_id, cat_id, feat_type, train_percent, use_whole_img)

        if examples == {}:
            return 0.0, 0.0, examples, num_pos, num_neg, mp
        
        cls = Classifier()
    cls.train(examples['train'], examples['train_lbls'])
    res = cls.test(examples['test'])

    if weighted_ap:
        ap_score = average_precision_score(examples['test_lbls'], res, average='weighted')
    else:
        ap_score = average_precision_score(examples['test_lbls'], res)
    acc = accuracy_score(examples['test_lbls'], [1 if x >=-0.0 else -1 for x in res])

    print 'AP score: %.4f' % ap_score
    print 'Accuracy (thresh = 0.0): %.4f' % acc

    cls.save(sname)

    return ap_score, acc, examples, num_pos, num_neg, mp 

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
        try:                    
            print Label.query.get(c.label_id).name
            print c.id
            mdl = joblib.load(c.location)
            conf = mdl.test(feat)
            res.append(conf)
        except (AttributeError, IOError), e:
            res.append(-10)
        
    return sorted(zip(attr_ids, res), key = lambda x: x[1], reverse=True)


def print_result(img, attr_confs, cat_id, sname):

    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')  # clear x- and y-axes
    if cat_id == -1:
        plt.title('all object classifier')
    else:
        plt.title(Label.query.get(cat_id).name)
    for ind, a in enumerate(attr_confs[:15]):
        attr = Label.query.get(a[0]).name
        t = '%s %0.3f' % (attr, a[1])
        print t
        plt.text(min(img.shape[1]+10, 1000), (ind+1)*img.shape[1]*0.1, t, ha='left')
    
    fig.savefig(sname, dpi = 300,  bbox_inches='tight')    
    pass    


def slugify(word):
    return word.replace('/', '_').replace(' ', '')

def get_multilabel(patch_id, obj_lbls=None, parent_lbls=None, cmplt=True):
    ann_vec = AnnotationVecMatView.query.filter(AnnotationVecMatView.patch_id == patch_id).first().vec
    if cmplt:
        multilabel = np.array([1 if x >= 0.5 else 0 for x in eval(ann_vec)])
    else:
        multilabel = np.array(eval(ann_vec))    
    if obj_lbls != None:
        obj_vec = sorted(set(obj_lbls + parent_lbls))
        a = Annotation.query.filter(Annotation.patch_id == patch_id).filter(Annotation.label_id.in_(obj_lbls)).first()
        obj_ann = np.zeros((len(obj_vec)))
        if a is not None:
            obj = a.label_id
            par = parent_lbls[obj_lbls.index(obj)]            
            obj_ann[obj_vec.index(obj)] = 1
            obj_ann[obj_vec.index(par)] = 1
            print 'obj lbl: {} {}'.format(par, obj)
        multilabel = np.hstack((multilabel, obj_ann))
    return multilabel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train an svm")
    parser.add_argument("--location", help="", type=str)        
    parser.add_argument("--feat_type", help="name of feature", type=str)    
    parser.add_argument("--train_percent", help="", type=float)
    parser.add_argument("--cat_id", help="", type=int)
    parser.add_argument("--label_id", help="", type=int)
    parser.add_argument("--whole_img", help="", type=bool, default=False)
    parser.add_argument("--weighted", help="", type=bool, default=False)

    args = parser.parse_args()

    # try:
    ap_score, acc, examples, num_pos, num_neg, mp = make_classifier(args.label_id, args.cat_id,
                                                                    args.feat_type, args.train_percent,
                                                                    args.location, args.whole_img, args.weighted)        
    c = ClassifierScore(type=args.feat_type,
                        location=args.location,
                        cat_id=args.cat_id,
                        label_id=args.label_id,
                        num_pos=num_pos,
                        num_neg=num_neg,
                        train_percent=args.train_percent,
                        test_acc=acc,
                        ap=ap_score)
    db.session.add(c)
    db.session.commit()
    # except ValueError, e:
    #     print 'model already trained.'
