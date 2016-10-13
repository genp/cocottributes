import json

from numpy import random as nprnd

from app import db
from app.models import *

def make_annotation_task(patch_ids, attr_ids, task_label, job_type='annotation'):
    '''
    Inputs:
    patch_ids - list of unique patch_ids length = 10 ideally for UI
    attr_ids - list of attributes to label (attribute_ids) length = 20 ideally for UI
    task_label - title for the category that is being labeled, e.g. cat, scene
    '''

    cmd = {}
    cmd['label'] = task_label

    attributes = []
    for id in attr_ids:
        name = Label.query.get(id).name
        attributes.append({'id': id, 'name': name})
    cmd['attributes'] = attributes
    
    patches = []
    images = []
    for patch_id in patch_ids:
        p = Patch.query.get(patch_id)
        seg = p.segmentation
        img_id = p.image_id
        images.append(img_id)

        patches.append({'id': patch_id, 'image_id': img_id, 'segmentation': str(seg)})
    cmd['patches'] = patches
    

    j = Jobs(cmd=json.dumps(cmd), job_type=job_type)
            
    db.session.add(j)
    db.session.commit()

    # todo make hit_details - single insert
    stmt = 'insert into hit_details (patch_id, image_id, label_id, num_hits, job_id) values '
    insert_vals = []
    for idx, patch_id in enumerate(patch_ids):
        for attr_id in attr_ids:
            insert_vals.append('(%d, %d, %d, 0, %d)' % (patch_id, images[idx], attr_id, j.id))

    print 'Creating %d instance labels for HIT %d...' % (len(insert_vals), j.id)
    stmt += ', '.join(insert_vals)
    db.engine.execute(stmt)


    return j.id

def make_annotation_all_imgs_task(patch_ids, attr_id, task_label, job_type='annotation'):
    '''
    Inputs:
    patch_ids - list of unique patch_ids length = 50-100 ideally for UI
    attr_id - attribute to label
    task_label - title for the category that is being labeled, e.g. cat, scene
    '''

    cmd = {}
    cmd['label'] = task_label

    name = Label.query.get(attr_id).name
    cmd['attribute'] = {'id': attr_id, 'name': name}
    
    patches = []
    images = []
    # make patches have x, y, w, h
    for patch_id in patch_ids:
        p = Patch.query.get(patch_id)
        seg = [json.loads(p.segmentation)[0]]
        segx = [seg[0][ix] for ix in range(0,len(seg[0]),2)]
        segy = [seg[0][iy] for iy in range(1,len(seg[0]),2)]
        img_id = p.image_id
        images.append(img_id)
        seg.append(p.x) 
        seg.append(p.y)
        seg.append(p.width)
        seg.append(p.height)
        img = Image.query.get(img_id)
        seg.append(img.width)
        seg.append(img.height)
        patches.append({'id': patch_id, 'image_id': img_id, 'segmentation': json.dumps(seg)})
    cmd['patches'] = patches
    

    j = Jobs(cmd=json.dumps(cmd), job_type=job_type)
            
    db.session.add(j)
    db.session.commit()

    # todo make hit_details - single insert
    stmt = 'insert into hit_details (patch_id, image_id, label_id, num_hits, job_id) values '
    insert_vals = []
    for idx, patch_id in enumerate(patch_ids):
        insert_vals.append('(%d, %d, %d, 0, %d)' % (patch_id, images[idx], attr_id, j.id))

    print 'Creating %d instance labels for HIT %d...' % (len(insert_vals), j.id)
    stmt += ', '.join(insert_vals)
    db.engine.execute(stmt)


    return j.id

def make_tasks(patch_ids, attr_ids, task_label, job_type='annotation', allimgs=False):
    '''
    Batches patch_ids into groups of 10 and attr_ids into groups of 20 
    Makes set of HITs
    allimgs=True for binary attribute labeling tasks with one image per query
    allimgs=False for multiple attribute annotation
    Output: list of job_ids created
    '''
    patch_group_size = 10
    if allimgs:
        patch_group_size = 50
    attr_group_size = 20

    jobs = []
    for p_idx in range(0, len(patch_ids), patch_group_size):
        sub_patch_ids = patch_ids[p_idx : min(p_idx + patch_group_size, len(patch_ids))]
        sub_patch_ids = fill_out_list(sub_patch_ids, patch_ids, patch_group_size, p_idx)
        if allimgs:
            new_job = make_annotation_all_imgs_task(sub_patch_ids, attr_ids, task_label, job_type) # attr_ids is really only one attr_id
            jobs.append(new_job)
        else:

            for a_idx in range(0, len(attr_ids), attr_group_size):
                sub_attr_ids = attr_ids[a_idx : min(a_idx + attr_group_size, len(attr_ids))]
                sub_attr_ids = fill_out_list(sub_attr_ids, attr_ids, attr_group_size, a_idx)
            
                new_job = make_annotation_task(sub_patch_ids, sub_attr_ids, task_label, job_type)
                jobs.append(new_job)
            print 'Made %d hits so far...' % len(jobs)
    
    print 'Jobs Created %d: ' % len(jobs)
    return jobs

def fill_out_list(sub_list, full_list, fill_size, max_val):
    '''
    Assumes full list is of unique values
    '''
    if len(sub_list) >= fill_size:
        return sub_list
    if fill_size - len(sub_list) > max_val:
        print "Asking to fill with fewer than nessessary values %d / %d " % (fill_size, max_val)
        return sub_list
    ret_list = set(sub_list)
    while len(ret_list) < fill_size:
        ret_list.add(full_list[nprnd.randint(max_val)])
    ret_list = list(ret_list)
    return ret_list
