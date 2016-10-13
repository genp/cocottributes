#!/usr/bin/env python
import argparse
import os

from app import db
from app.models import *
from mturk import manage_hits

def create_bboxes(save_dir, cat_lbls, attr_lbls):
    img_dir = os.path.join(save_dir, 'images')
    pids = []
    cats = []
    for cat in cat_lbls:
        tmppids = manage_hits.find_patches([cat]+attr_lbls, [])
        pids += tmppids
        print '%d instances of category %d' % (len(tmppids), cat)
        cats += [cat for x in range(len(tmppids))]
    resize = (256, 256)
    ftest_cat = open(os.path.join(save_dir,'test.txt'), 'w')
    ftest_attr = open(os.path.join(save_dir,'test_attr.txt'), 'w')
    ftrain_cat = open(os.path.join(save_dir,'train.txt'), 'w')
    ftrain_attr = open(os.path.join(save_dir,'train_attr.txt'), 'w')

    final = len(pids)    
    for ind, pid in enumerate(pids):
        print '%d of %d...' % (ind, final)
        p = Patch.query.get(pid)
        img_name = os.path.basename(p.image.location)
        savename = os.path.join(img_dir, img_name)
        p.crop(savename=savename, make_square=True, resize=resize)
        attrs = [ann.label_id for ann in p.annotations.all() if ann.value == True and ann.label_id in attr_lbls]
        if ind % 5 == 0:
            ftest_cat.write('%s %d\n' % (savename, cats[ind]))
            for attr in attrs:
                ftest_attr.write('%s %d\n' % (savename, attr))
        else:
            ftrain_cat.write('%s %d\n' % (savename, cats[ind]))
            for attr in attrs:
                ftrain_attr.write('%s %d\n' % (savename, attr))
                
    ftest_cat.close(); ftest_attr.close(); ftrain_cat.close(); ftrain_attr.close();
        
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='add stuff to cocottributes database')
#     parser.add_argument('save_dir', type=str, 
#                         help='location to save bbox images and test.txt and train.txt')


#     args = parser.parse_args()
#     create_bboxes(args.save_dir)
