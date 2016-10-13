import os
import json
import datetime

from app import db
from app.models import *
import utils
import config

def load_coco(ann_dir):
    '''
    Loads coco schema into postgres database
    Expects a fresh, empty db
    '''
    worker = Worker(username = 'coco2014', password = '')
    db.session.add(worker)
    hit_resp = HitResponse(time = 0, confidence = 5, worker = worker)
    db.session.add(hit_resp)
    db.session.commit()    
    data_types = ['val2014', 'train2014']
    for dt in data_types:
        annFile=os.path.join(ann_dir, 'instances_%s.json'%(dt))

        with open(annFile, 'r') as f:
            data = json.load(f)
            
        if dt == 'val2014':
            load_cats(data['categories'])

        load_imgs(data['images'], dt)

        load_ann(data['annotations'])

def load_cats(data):
    # add categories to table Label
    stmt = 'insert into label values '
    cnt = 0

    for d in data:                    
        if cnt > 0:
            stmt += ', '
        stmt += "(%d, '%s')" % (d['id'], d['name'])
        cnt += 1
    db.engine.execute(stmt)
    update_id_seq('label')
    db.session.commit()
    
    # add references to supercategories
    for d in data:
        cat = Label.query.get(d['id'])
        parent = Label.query.filter(Label.name == d['supercategory']).first()
        if parent == None:
            parent = Label(name = d['supercategory'], parent_id=None)
            db.session.add(parent)
            db.session.commit()
        cat.parent_id = parent.id
        db.session.commit()

def load_imgs(data, dt):
    # add images to table Image
    stmt = 'insert into image values '
    cnt = 0

    for d in data:                    
        if cnt > 0:
            stmt += ', '
        id = d['id']
        location = d['file_name']
        url = d['url']
        width = d['width']
        height = d['height']
        ext = os.path.splitext(location)[1]
        try:
            mime = config.mime_dictionary[ext]
        except KeyError, e:
            mime = ''

        latitude, longitude = (-1, -1)
        
        stmt += "(%d, '%s', '%s', '%s', '%s', '%s', %f, %f, %d, %d)" % (id, ext, mime, location, dt, url, latitude, longitude, width, height)
        cnt += 1
        if cnt > 1000:
            db.engine.execute(stmt)
            stmt = 'insert into image values '
            cnt = 0
    db.engine.execute(stmt)
    update_id_seq('image')
    db.session.commit()

def load_ann(data):
    # add annotations to table Annotation
    for d in data:
        image_id = d['image_id']
        x = d['bbox'][0]
        y = d['bbox'][1]
        width = d['bbox'][2]
        height = d['bbox'][3]
        segmentation = str(d['segmentation']).replace("'", "''")
        area = d['area']
        is_crowd = True if d['iscrowd'] == 1 else False
        patch_stmt = "insert into patch values (default, %f, %f, %f, %f, '', '%s', %f, %r, %d) returning id" % (x, y, width, height, segmentation, area, is_crowd, image_id)
        patch_id = db.engine.execute(patch_stmt).fetchall()[0][0]
        id = d['id']
        timestamp = str(datetime.datetime.now())
        
        label_id = d['category_id']
        hit_id = utils.get_first_db_res("select h.id from hit_response h, worker w where w.id = h.worker_id and w.username = 'coco2014'")
        value = True
        ann_stmt = "insert into annotation values (%d, %r, '%s', %d, %d, %d, %d)" % (id, value, timestamp, patch_id, image_id, label_id, hit_id)
        db.engine.execute(ann_stmt)

    update_id_seq('annotation')
    db.session.commit()


def update_id_seq(table_name):
    nexval = db.engine.execute('select max(id) from '+table_name).fetchall()[0][0]+1
    db.engine.execute('alter sequence %s_id_seq restart with %d' %(table_name, nexval))                      
    
