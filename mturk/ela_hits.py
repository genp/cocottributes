#!/usr/bin/env python
import argparse
import datetime
from collections import defaultdict

import numpy as np
from sklearn.externals import joblib
from sqlalchemy import or_

from app import db
from app.models import * 
from mturk import manage_hits
from mturk import make_task
from semantic_graph import cooccurrence

''' limit to number of queries per patch '''
NUMQ = {}
for parent in [1, 91, 93, 97]:
    for lbl in Label.query.filter(Label.parent_id == parent).all():
        if lbl.parent_id == 1:
            NUMQ[lbl.id] = 70
        elif lbl.parent_id == 91:
            NUMQ[lbl.id] = 20
        elif lbl.parent_id == 93:
            NUMQ[lbl.id] = 50
        elif lbl.parent_id == 97:
            NUMQ[lbl.id] = 15
        else:
            NUMQ[lbl.id] = 40

''' number of patches per hit '''
HITNUM = 50
''' file name for hierarchical label sublist '''
SUBLIST_FNAME = '/root/coco_attributes/data/%s_attr_sublist.jbl'
''' file name for hierarchical labeled training sets '''
SET_FNAME = '/root/coco_attributes/data/%s_exhaustive_labeled_set.jbl'

def get_lbl_names():
    stmt = 'select id, name from label'
    lbl_items = db.engine.execute(stmt).fetchall()
    lbl_names = defaultdict(int)
    for item in lbl_items:
        lbl_names[item[0]] = item[1]
    return lbl_names        

_LBL_NAMES = get_lbl_names()

# func for launching new hits from that table and marking launched queries
def launch_query_hits(label_id, cat_id, clean_up=False):
    queries = Query.query.filter(Query.label_id == label_id).filter(Query.cat_id == cat_id).filter(Query.hit_launched == False).all()
    subind = len(queries) % HITNUM 
    if subind > 0 and not clean_up:
        queries = queries[:-1*subind]
    patch_ids = [x.patch_id for x in queries]
    cat = Label.query.get(cat_id)
    # make allimgs tasks for patch_ids, attr_id
    job_type = 'ela_hit'
    task_label = cat.name
    jobs = make_task.make_tasks(patch_ids, label_id, task_label, job_type, allimgs=True)

    # launch jobs
    task_file = '/root/coco_attributes/aws-mturk-clt-1.3.1/annotation_all_imgs_task/annotation_all_imgs.input'
    mturk_rel_path = '../annotation_all_imgs_task/annotation_all_imgs'
    manage_hits.launch_hits(jobs, task_file, mturk_rel_path)

    for q in queries:
        q.hit_launched = True
    db.session.commit()
    
    return jobs


def schedule_next_query(patch_id, cat_id):
    '''
    func for adding (patch, attribute) question to Query table, 
    checks if enough queries to launch new hit, and if so does
    '''
    cat = Label.query.get(cat_id)
    # get sublist for this parent type
    label_ids = joblib.load(SUBLIST_FNAME % Label.query.get(cat.parent_id).name)
    # get annotation vector
    cur_label, ann_cnt = Patch.query.get(patch_id).annotation_vector(label_ids, consensus=True)
    
    num_labeled = len([x for x in cur_label if x > -1])
    if num_labeled >= NUMQ[cat_id]:
        print 'labeled up to limit'
        return -1
    
    # get training set with parent type
    # TODO: change to include category label in attr feature vectors
    labels_train = joblib.load(SET_FNAME % Label.query.get(cat.parent_id).name)['attributes']
    print 'calculating next query'
    sgraph = cooccurrence.SGraph(train=labels_train, 
                                 dtree={}, 
                                 ela_type='dist', 
                                 ela_limit_type = 'numq', 
                                 ela_limit = NUMQ[cat_id])

    
    query_ind = sgraph.get_next_query(cur_label)[0]
    print 'Next question for %d about %s...' % (patch_id, _LBL_NAMES[label_ids[query_ind]])
    check = len(Query.query.\
                   filter(Query.label_id == label_ids[query_ind]).\
                   filter(Query.cat_id == cat_id).\
                   filter(Query.patch_id == patch_id).all())
    if check > 0:
        print 'already added this query'
        return 0
    q = Query(patch_id = patch_id, cat_id = cat_id, label_id = label_ids[query_ind])
    db.session.add(q)
    db.session.commit()
    # func to check if enough annotations exist to calc next query for patch
    queries = Query.query.\
                   filter(Query.label_id == label_ids[query_ind]).\
                   filter(Query.cat_id == cat_id).\
                   filter(Query.hit_launched == False).all()
    num_queries = len(queries)
    if num_queries >= HITNUM:
        launch_query_hits(label_ids[query_ind], cat_id)

    return label_ids[query_ind]


def get_waiting_queries():
    stmt = "select distinct (cat_id, label_id), count(*), max(timestamp) from query where hit_launched is false group by (cat_id, label_id) order by count(*) desc"
    res = db.engine.execute(stmt).fetchall()
    return res

def clean_up_waiting_queries(interval = datetime.timedelta(days=7)):
    query_pairs = get_waiting_queries()
    jobs = []
    for item in query_pairs:
        if  datetime.datetime.now() - interval > item[2]:
            pair = eval(item[0])
            cat_id = pair[0]
            label_id = pair[1]
            j = launch_query_hits(label_id, cat_id, clean_up=True)
            jobs += j
    return jobs



def launch_fresh_patches(cats, limit):
    '''
    cats is a list of Label ids for which to launch new ELA labeling jobs
    '''
    without_lbls_file = 'data/%s_attr_sublist.jbl'

    for cat in cats:
        without_lbls = joblib.load(without_lbls_file % Label.query.get(Label.query.get(cat).parent_id).name)
        with_lbls = [cat]
        
        # get labeled set
        print 'finding training patches'
        patch_ids = manage_hits.find_patches(with_lbls+without_lbls, [], [])
        labels_train = manage_hits.labeled_set(patch_ids, without_lbls)
        print 'training set size '+str(labels_train['attributes'].shape)
        if labels_train['attributes'].shape[0] == 0:
            continue
        # get  first step ela for each category
        print 'calculating next query'
        sgraph = cooccurrence.SGraph(train=labels_train['attributes'], 
                                     dtree={}, 
                                     ela_type='dist', 
                                     ela_limit_type = 'numq', 
                                     ela_limit =20)
        cur_label = -1*np.ones((len(without_lbls), 1))
        query_ind = sgraph.get_next_query(cur_label)[0]
        # print 'Next question for %s about %s...' % (Label.query.get(cat).name, Label.query.get(without_lbls[query_ind]).name)

        # select limit number of  unlabeld patches
        unlabeled_pids = manage_hits.find_patches(with_lbls, without_lbls, [])[:limit]

        # make allimgs tasks for patch_ids, attr_id    
        job_type = '%s_%s_v1' % (Label.query.get(cat).name, Label.query.get(Label.query.get(cat).parent_id).name)
        task_label = Label.query.get(cat).name
        jobs = make_task.make_tasks(unlabeled_pids, without_lbls[query_ind], task_label, job_type, allimgs=True)
        
        # input('Jobs for %s : %s ' % (Label.query.get(cat), str(jobs[:5])))

        # launch jobs
        task_file = '/root/coco_attributes/aws-mturk-clt-1.3.1/annotation_all_imgs_task/annotation_all_imgs.input'
        mturk_rel_path = '../annotation_all_imgs_task/annotation_all_imgs'
        manage_hits.launch_hits(jobs, task_file, mturk_rel_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--categories", type=int, nargs='+', help='categories to be labeled using the ELA')
    parser.add_argument("-l", "--limit", type=int, help='number of patches to be launched per category')
    args = parser.parse_args()
    launch_fresh_patches(args.categories, args.limit)
