#!/usr/bin/env python
from app import db
from app.models import *

import print_dataset


if __name__ == "__main__":
    # check consensus omits blocked workers
    # check annotation votes
    orig_ann_id = 431265
    label_id = 418
    # patch_id = Annotation.query.get(orig_ann_id).patch_id
    # anns = [x[0].value for x in \
    #          db.session.query(Annotation, Label, HitResponse, Worker)\
    #                    .join(Label).join(HitResponse).join(Worker)\
    #                    .filter(Annotation.patch_id == patch_id)\
    #                    .filter(Label.id == label_id).all()]
    # print len(anns)    
    # print print_dataset.consensus_label(orig_ann_id, label_id)
    label_type = 'object attributes'
    # parent_lbl = Label.query.filter(Label.name == label_type).first()
    
    # print print_dataset.get_categories(parent_lbl)    
    # print len(print_dataset.get_item_anns(orig_ann_id, cat_parent_id=407))


    # check annotation format

    outfile = '/home/gen/scratch/cocottributes.json'
    print_dataset.get_annotations(label_type, outfile)
