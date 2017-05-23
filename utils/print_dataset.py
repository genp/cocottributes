#!/usr/bin/env python
from collections import defaultdict
import argparse
import datetime
import json
import logging
import os

from sklearn.externals import joblib

from app import db
from app.models import *
import config

'''
Script to print cocottributes dataset to MS COCO dataset json format.

keys = ['info', 'type', 'annotations', 'categories']      
'''

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def metadata(data):
    '''
    input: dictionary <data>
    output: dictionary <data> updated to include
            information about the dataset contained in <data>
    '''

    data['info'] = {'contributor': 'MS COCO Attributes group',
                    'date_created': str(datetime.datetime.now()),
                    'description': '2016 MS COCO Attributes dataset',
                    'url': 'http://cocottributes.org',
                    'version': '0.5',
                    'year': 2016}
    data['type'] = 'attributes'
    return data
    
def get_annotations(label_type, outfile):
    '''
    get all annotations for labels with parent_label that has name <label_type>
    '''
    # Dictionary containing dataset
    data = {}

    # Metadata for dataset
    data = metadata(data)

    # Annotation categories
    parent_lbl = Label.query.filter(Label.name == label_type).first()
    data['attributes'] = get_categories(parent_lbl)
    logging.debug('Number of attributes: %d' % len(data['attributes']))
    

    # Annotations by average worker vote
    #data['annotations'], data['ann_vecs'], data['patch_id_to_ann_id'] = get_votes(parent_lbl)
    data['ann_vecs'], data['patch_id_to_ann_id'] = get_votes(parent_lbl)
    data['split'] = get_split_types(data['ann_vecs'].keys())
    logging.debug('Number of annotations: %d' % len(data['ann_vecs']))
    
    joblib.dump(data, outfile, compress = 6)
    # with open(outfile, 'w') as f:
    #     json.dump(data, f)

    return

def get_categories(parent_lbl):
    # assumes unique super category label

    categories = Label.query.filter(Label.parent_id == parent_lbl.id)
    cat_fmt = []
    for cat in categories:
        tmp = {}
        tmp['id'] = cat.id
        tmp['name'] = cat.name
        tmp['supercategory'] = parent_lbl.name
        cat_fmt.append(tmp)
    return cat_fmt


def get_votes(parent_lbl):

    # Pull all annotations for labels with given supercategory
    # No blocked workers
    # format [(patch_id, label_id, avg_vote, num_votes)]
    stmt = ('select a.patch_id, a.label_id, sum(cast(value as int))/cast(count(*) as float), count(*) '
            'from annotation a, hit_response hr, worker w '
            'where a.hit_id = hr.id and hr.worker_id = w.id and w.is_blocked is False '
            'and a.label_id in (select id from label where parent_id = %d) group by a.patch_id, a.label_id'
            ) % parent_lbl.id
    all_anns = db.engine.execute(stmt).fetchall()
    attrs_ordered = [x.id for x in Label.query.filter(Label.parent_id == parent_lbl.id).order_by(Label.id).all()]
    # Get all original MS COCO annotated instances
    stmt = 'select a.patch_id, a.id from annotation a where a.hit_id = 1'
    coco_items = db.engine.execute(stmt).fetchall()
    coco_ann_ids = defaultdict(int)
    for item in coco_items:
        coco_ann_ids[item[0]] += item[1]

    ann_vecs = {}
    # Format annotations to match up with COCO annotation ids    
    # fmt_anns = []
    for ann in all_anns:
        # consensus_ann = {}
        # consensus_ann['coco_annotation_id'] = coco_ann_ids[ann[0]]
        # consensus_ann['attr_id'] = ann[1]
        # consensus_ann['value'] = ann[2]        
        # consensus_ann['num_votes'] = ann[3]
        # fmt_anns.append(consensus_ann)
        try:
            ann_vecs[ann[0]][attrs_ordered.index(ann[1])] = ann[2]
        except KeyError, e:
            ann_vecs[ann[0]] = -1*np.ones(len(attrs_ordered))
            ann_vecs[ann[0]][attrs_ordered.index(ann[1])] = ann[2] if ann[3] > -0.01 else -1

    

    return ann_vecs, coco_ann_ids#fmt_anns, ann_vecs, coco_ann_ids

def get_split_types(patch_ids):
    patch_split = {}
    for p in patch_ids:
        patch_split[p] = Patch.query.get(p).image.type

    return patch_split

def materialize_view(parent_lbl):
    # Populate a materialized view type table with the annotation vector for a given patch for corresponding label ids
    # unlabed elements marked with -1
    # Pull all annotations for labels with given supercategory
    # No blocked workers
    # format [(patch_id, label_id, avg_vote, num_votes)]
    stmt = ('select a.patch_id, a.label_id, sum(cast(value as int))/cast(count(*) as float), count(*) '
            'from annotation a, hit_response hr, worker w '
            'where a.hit_id = hr.id and hr.worker_id = w.id and w.is_blocked is False '
            'and a.label_id in (select id from label where parent_id = %d) group by a.patch_id, a.label_id'
            ) % parent_lbl.id
    all_anns = db.engine.execute(stmt).fetchall()
    attrs_ordered = [x.id for x in Label.query.filter(Label.parent_id == parent_lbl.id).order_by(Label.id).all()]

    updated_patches = set()
    print "Num annotations: {}".format(len(all_anns))
    for ann in all_anns:

        cur_vec = AnnotationVecMatView.query.filter(AnnotationVecMatView.patch_id == ann[0])\
                                            .filter(AnnotationVecMatView.label_ids == str(attrs_ordered))\
                                            .first()
        if cur_vec == None:
            vec = [-1 for x in range(len(attrs_ordered))]
            cur_vec = AnnotationVecMatView(patch_id = ann[0], label_ids = str(attrs_ordered), vec = str(vec))
            db.session.add(cur_vec)
        else:
            vec = eval(cur_vec.vec)

        vec[attrs_ordered.index(ann[1])] = float("{0:.3f}".format(ann[2])) if ann[3] > 0 else -1
        cur_vec.vec = str(vec)
        db.session.commit() 
        updated_patches.add(ann[0])
        print cur_vec.id

    return len(updated_patches)

def stringify_vec(vec):
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_file", type=str, help='file to save to (include .jbl suffix)')
    args = parser.parse_args()

    get_annotations(label_type='object attributes', outfile=args.save_file)
