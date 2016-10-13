import json

from app import db
from app.models import *
from utils import utils
# turn annotation labels by hit X into a quiz Job
def annotation_to_quiz(hit_id, alt_hit_id, quiz_label):
    '''
    hit_id and alt_hit_id should be for the same task. hit_id has the strictly correct answers and alt_hit_id has possibly correct.
    '''
    anns = utils.get_all_db_res('select value, patch_id, image_id, label_id from annotation where hit_id = %d' % hit_id)
    cmd = {}
    cmd['label'] = quiz_label
    values, patch_ids, image_ids, label_ids = zip(*anns)

    attr_ids = sorted(set(label_ids), key=lambda x: label_ids.index(x))
    attributes = []
    for id in attr_ids:
        name = Label.query.get(id).name
        attributes.append({'id': id, 'name': name})
    cmd['attributes'] = attributes
    
    unique_patch_ids = sorted(set(patch_ids), key=lambda x: patch_ids.index(x))

    patches = []
    for patch_id in unique_patch_ids:
        p = Patch.query.get(patch_id)
        seg = p.segmentation
        img_id = p.image_id
        
        patches.append({'id': patch_id, 'image_id': img_id, 'segmentation': str(seg)})
    cmd['patches'] = patches
    
    answers = {}

    for idx, val in enumerate(values):
        try:
            cur_dict = answers[str(patch_ids[idx])]
        except KeyError, e:
            answers[str(patch_ids[idx])] = {}
            cur_dict = answers[str(patch_ids[idx])]

        cur_dict[str(label_ids[idx])] = 1 if val else 0

    cmd['answers'] = answers

    alt_anns = utils.get_all_db_res('select value, patch_id, image_id, label_id from annotation where hit_id = %d' % alt_hit_id)
    values, patch_ids, image_ids, label_ids = zip(*alt_anns)
    alt_answers = {}
    for idx, val in enumerate(values):
        try:
            cur_dict = alt_answers[str(patch_ids[idx])]
        except KeyError, e:
            alt_answers[str(patch_ids[idx])] = {}
            cur_dict = alt_answers[str(patch_ids[idx])]

        cur_dict[str(label_ids[idx])] = 1 if val else 0
    cmd['alt_answers'] = alt_answers
    j = Jobs(cmd=json.dumps(cmd), job_type='quiz')
            
    db.session.add(j)
    db.session.commit()
    return j.id

def allimgs_annotation_to_quiz(hit_id, alt_hit_id, quiz_label):
    '''
    hit_id and alt_hit_id should be for the same task. hit_id has the strictly correct answers and alt_hit_id has possibly correct.
    '''
    anns = utils.get_all_db_res('select value, patch_id, image_id, label_id from annotation where hit_id = %d' % hit_id)
    cmd = {}
    cmd['label'] = quiz_label
    values, patch_ids, image_ids, label_ids = zip(*anns)

    attr_id = label_ids[0]
    name = Label.query.get(attr_id).name
    attribute = {'id':attr_id, 'name': name}
    cmd['attribute'] = attribute
    
    unique_patch_ids = sorted(set(patch_ids), key=lambda x: patch_ids.index(x))

    patches = []
    # make patches have x, y, w, h
    for patch_id in patch_ids:
        p = Patch.query.get(patch_id)
        seg = [json.loads(p.segmentation)[0]]
        segx = [seg[0][ix] for ix in range(0,len(seg[0]),2)]
        segy = [seg[0][iy] for iy in range(1,len(seg[0]),2)]
        img_id = p.image_id
        seg.append(p.x) 
        seg.append(p.y)
        seg.append(p.width)
        seg.append(p.height)
        img = Image.query.get(img_id)
        seg.append(img.width)
        seg.append(img.height)
        patches.append({'id': patch_id, 'image_id': img_id, 'segmentation': json.dumps(seg)})
    cmd['patches'] = patches

    
    answers = {}

    for idx, val in enumerate(values):
        try:
            cur_dict = answers[str(patch_ids[idx])]
        except KeyError, e:
            answers[str(patch_ids[idx])] = {}
            cur_dict = answers[str(patch_ids[idx])]

        cur_dict[attr_id] = 1 if val else 0

    cmd['answers'] = answers

    alt_anns = utils.get_all_db_res('select value, patch_id, image_id, label_id from annotation where hit_id = %d' % alt_hit_id)
    values, patch_ids, image_ids, label_ids = zip(*alt_anns)
    attr_id = label_ids[0]
    alt_answers = {}
    for idx, val in enumerate(values):
        try:
            cur_dict = alt_answers[str(patch_ids[idx])]
        except KeyError, e:
            alt_answers[str(patch_ids[idx])] = {}
            cur_dict = alt_answers[str(patch_ids[idx])]

        cur_dict[attr_id] = 1 if val else 0
    cmd['alt_answers'] = alt_answers
    j = Jobs(cmd=json.dumps(cmd), job_type='quiz')
            
    db.session.add(j)
    db.session.commit()
    return j.id

