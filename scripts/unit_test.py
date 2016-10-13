#!/usr/bin/env python

# from app.models import *
# from app import db
# lbls = [x.id for x in db.session.query(Label).filter(Label.parent_id == 407).order_by(Label.id)]
# p = Patch.query.get(138058)
# missing = p.missing_labels(lbls)
# print missing
# print len(missing)

# from mturk import manage_hits
# job_type = 'food_annotation_exhaustive_pizza_v1'
# missing_jobs = manage_hits.get_missing_hits(job_type)
# print len(missing_jobs)
# print manage_hits.launch_missing_hits(job_type, 'aws-mturk-clt-1.3.1/annotation_single_task/annotation_single.input', '../annotation_single_task/annotation_single')

import numpy
from app import db
from app.models import *
from mturk import manage_hits

attr_lbls = [x.id for x in db.session.query(Label).filter(Label.parent_id == 407).order_by(Label.id)]
# pids = manage_hits.find_patches([18], attr_lbls)
# print len(pids)
# fully labeled person set
pids = manage_hits.find_patches([1]+attr_lbls, [])
print pids
print len(pids)
lbl_set = manage_hits.labeled_set(pids, attr_lbls)
print lbl_set.shape
