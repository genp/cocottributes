# import some modules
import sys, os, time
import os.path as osp
sys.path.append('../')

import numpy as np
import scipy

from app import db
from app.models import *

# load exhaustive set
stmt = "select patch_id from (select a.patch_id, count(distinct label_id) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count > 175"
patch_ids_train = [x[0] for x in db.engine.execute(stmt).fetchall()]

# load partially annotated set (those that have more than 20 labels, unlabeled attrs marked negative)
stmt = "select patch_id from (select a.patch_id, count(distinct label_id) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count >= 20 and count <= 175"
patch_ids_val = [x[0] for x in db.engine.execute(stmt).fetchall()]

rand_inds = list(set(np.random.randint(len(patch_ids_val), size=7500)))[:6500]
patch_ids_rand_val = [patch_ids_val[x] for x in rand_inds]

# load partially annotated set (those that have more than 20 labels, unlabeled attrs marked negative)
stmt = "select patch_id from (select a.patch_id, count(distinct label_id) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count >= 40 and count <= 175"
patch_ids_val2 = [x[0] for x in db.engine.execute(stmt).fetchall()]

N = len(patch_ids_train)
M = len(patch_ids_val)
print 'Train imgs:{}'.format(N)
print 'Val imgs:{}'.format(M)

def get_multilabel(patch_id):
    ann_vec = AnnotationVecMatView.query.filter(AnnotationVecMatView.patch_id == patch_id).first().vec
    multilabel = np.array([1 if x >= 0.5 else 0 for x in eval(ann_vec)])
    return multilabel

ex_set = np.zeros((N, 204))
for in_idx, in_ in enumerate(patch_ids_train):
    # load labels:
    ex_set[in_idx, :] = np.array(get_multilabel(in_)) 

part_set = np.zeros((M, 204))
for in_idx, in_ in enumerate(patch_ids_val):
    # load labels:
    part_set[in_idx, :] = np.array(get_multilabel(in_)) 

part_set_rand = np.zeros((M, 204))
for in_idx, in_ in enumerate(patch_ids_rand_val):
    # load labels:
    part_set_rand[in_idx, :] = np.array(get_multilabel(in_))

part_set2 = np.zeros((M, 204))
for in_idx, in_ in enumerate(patch_ids_val2):
    # load labels:
    part_set2[in_idx, :] = np.array(get_multilabel(in_)) 

# TODO: how to calculate this?? -- make a measure of how varied the fully annotated from the partially annotated dataset 


# for each partially annotated instance, calculate 'surprisingness' - min hamming distance from any item in exhaustive dataset
# print avg min hamming distance of quartiles
def find_min_h_dist(refset, testset, same=False):
    min_dist = []
    for idx, item in enumerate(testset):
        minhd = 1000
        if same:
            refsetp = refset
        else:
            refsetp = np.vstack((refset[:idx,:], refset[idx+1:, :]))
        for refitem in refsetp:
            hd = scipy.spatial.distance.hamming(item, refitem)
            if hd < minhd:
                minhd = hd
        min_dist.append(minhd)
    return min_dist

min_h_dist = {}
min_h_dist['ex'] = find_min_h_dist(ex_set, ex_set)
min_h_dist['ex_patch_ids'] = patch_ids_train
min_h_dist['part'] = find_min_h_dist(ex_set, part_set)
min_h_dist['part_patch_ids'] = patch_ids_val
min_h_dist['part_rand'] = find_min_h_dist(ex_set, part_set_rand)
min_h_dist['part_patch_ids_rand'] = patch_ids_rand_val
min_h_dist['part40'] = find_min_h_dist(ex_set, part_set2)
min_h_dist['part_patch_ids40'] = patch_ids_val2

print [x*204 for x in mquantiles(min_h_dist['part_rand'], [.1, .2, .3, .4, .5, .6, .7, .8, .9])]

joblib.dump(min_h_dist, 'ex_vs_part_h_dist_03032016.jbl')


def find_min_h_item(refset, testitem):

    minhd = 1000
    idx = -1
    for in_,refitem in enumerate(refset):
        hd = scipy.spatial.distance.hamming(testitem, refitem)
        if hd < minhd:
            minhd = hd
            idx = in_

    return minhd, idx
