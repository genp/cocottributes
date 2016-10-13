#!/usr/bin/env python
coco_root = '/data/hays_lab/COCO/coco/'
cocottributes_root = '/home/gen/coco_attributes/'
# import some modules
import sys, os, time
import os.path as osp
caffe_root = '/home/gen/caffe/'  
sys.path.append(caffe_root+'python')
sys.path.append(cocottributes_root+'caffe/')
sys.path.append(cocottributes_root)
import numpy as np
import lmdb
import caffe

from app import db
from app.models import *

im_size = [227, 227]
label_ids = [x.id for x in Label.query.filter(Label.parent_id == 407).order_by(Label.id).all()]
def load_cocottributes_patch(patch_id):
    """
    Load MS COCO cropped patch instance given AnnotationVecMatView item
    """
    patch= Patch.query.get(patch_id)
    im = patch.crop(savename = None, make_square = True, resize = im_size)
    return im


def get_multilabel(patch_id, obj_lbls=None, parent_lbls=None, cmplt=True):

    ann_vec = AnnotationVecMatView.query.filter(AnnotationVecMatView.patch_id == patch_id).first().vec
    # except AttributeError, e:
    #     print 'no mat view for {}'.format(patch_id)
    #     patch = Patch.query.get(patch_id)
    #     ann_vec,_ = patch.annotation_vector(label_ids, consensus=True)
    #     ann_vec = list(ann_vec)
    #     cur_vec = AnnotationVecMatView(patch_id = patch_id, label_ids = str(label_ids), vec = str(ann_vec))
    #     db.session.add(cur_vec)
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
        print multilabel.shape
    return multilabel

# Using instances with at least 20 labels
stmt = "select * from (select patch_id from (select a.patch_id, count(*) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count > 60 ) as foo intersect select p.id from patch p, image im where p.image_id = im.id and im.type = 'train2014'"
patch_ids_train = [x[0] for x in db.engine.execute(stmt).fetchall()]
stmt = "select * from (select patch_id from (select a.patch_id, count(*) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count > 60 ) as foo intersect select p.id from patch p, image im where p.image_id = im.id and im.type = 'val2014'"
patch_ids_val = [x[0] for x in db.engine.execute(stmt).fetchall()]

N = len(patch_ids_train)
print 'Train imgs:{}'.format(N)
print 'Val imgs:{}'.format(len(patch_ids_val))


save_path = '/data/gen_data/COCO/caffemodels_w_obj'
full_obj_lbls = Label.query.filter(Label.parent_id.in_([1,91,93,97])).order_by(Label.id).all()
obj_lbls = [x.id for x in full_obj_lbls]
parent_lbls = [x.parent_id for x in full_obj_lbls]

X = np.zeros((N, 3, 227, 227), dtype=np.uint8)
if not obj_lbls:
    y = np.zeros((N, 204), dtype=np.uint8)
else:
    adtl_lbls = len(obj_lbls)+len(set(parent_lbls))
    y = np.zeros((N, 204+adtl_lbls), dtype=np.uint8)    

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.

map_size = y.nbytes * 100
in_db = lmdb.open(osp.join(save_path, 'train-label-lmdb'), map_size=map_size)
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(patch_ids_train):
        # load labels:
        im = np.array(get_multilabel(in_, obj_lbls, parent_lbls))# or load whatever ndarray you need
        im = im.reshape(list(im.shape)+[1,1]) 
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
        print 'train lbl:{}'.format(in_idx)
in_db.close()
in_db = lmdb.open(osp.join(save_path, 'val-label-lmdb'), map_size=map_size)
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(patch_ids_val):
        # load labels:
        im = np.array(get_multilabel(in_, obj_lbls, parent_lbls)) # or load whatever ndarray you need
        im = im.reshape(list(im.shape)+[1,1])                
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
        print 'val lbl:{}'.format(in_idx)
in_db.close()

map_size = X.nbytes * 100
in_db = lmdb.open(osp.join(save_path, 'train-image-lmdb'), map_size=map_size)
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(patch_ids_train):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(load_cocottributes_patch(in_)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
        print 'train im:{}'.format(in_idx)
in_db.close()
in_db = lmdb.open(osp.join(save_path, 'val-image-lmdb'), map_size=map_size)
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(patch_ids_val):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(load_cocottributes_patch(in_)) # or load whatever ndarray you need        
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
        print 'val im:{}'.format(in_idx)
in_db.close()
