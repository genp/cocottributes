import json
import os
import subprocess

import numpy as np
import shutil

import config
from app import app, db
from app.models import * 

def get_recent_job_types(interval):
    '''
    interval is a string, e.g. '1 month' or '12 hours'
    '''
    job_types = [x[0] for x in db.engine.execute("select distinct job_type, max(start_time) from jobs where start_time > current_timestamp - interval '%s' group by job_type order by max(start_time) desc" % interval).fetchall()]

    return job_types

# finds whole hits that need to be repeated
def get_missing_hits(job_type):
    '''
    returns list of job_ids that have less than 3 hits by trusted workers
    job_id occur multiple times if missing multiple hits
    '''
    stmt = "select id, cnt from (select j.id, count(*) as cnt from jobs j, hit_response h where h.job_id = j.id and j.job_type = '%s' group by j.id) as hit_cnt where cnt < 3" % job_type    
    job_cnt = db.engine.execute(stmt).fetchall()

    jobs_to_print = []

    for item in job_cnt:
        for cnt in range(int(item[1])):
            jobs_to_print.append(item[0])

    return jobs_to_print

def launch_missing_hits(job_type, task_file, mturk_rel_path):
    jobs = get_missing_hits(job_type)
    return launch_hits(jobs, task_file, mturk_rel_path)
    

def launch_hits(jobs, task_file, mturk_rel_path):
    '''
    list of jobs to be launched on mturk
    task_file to write jobs to
    mturk_rel_path - task dir argument to run.sh
    '''
    f = open(task_file, 'w')
    f.writelines('job_id\n')
    f.write('\n'.join([str(x) for x in jobs]))
    f.close()
    cwd = os.getcwd()
    os.chdir(config.mturk_bin_path)
    output = subprocess.check_output("./run.sh %s" % (mturk_rel_path), shell=True)
    # appends hit ids to long running file
    os.system("touch %s" % task_file.replace('input', 'success_all'))
    output = subprocess.check_output("cat %s >> %s" % (task_file.replace('input', 'success'), task_file.replace('input', 'success_all')), 
                                     shell=True)
    os.chdir(cwd)
    return output


def labeled_set(patch_ids, label_ids):
    '''
    turns set of patches and set of label_ids into a matrix (n patches x m labels) stored in labels['attributes']
    labels['category'] includes  encoded vector of category and heirarchy membership
    '''
    attrs = np.zeros((len(patch_ids), len(label_ids)))
    cat_tuples = []
    cat_set = set()
    for p_idx, p in enumerate(patch_ids):
        patch = Patch.query.get(p)
        ann_vec, ann_cnt_vec = patch.annotation_vector(label_ids, consensus = True)
        attrs[p_idx,:] = ann_vec.reshape(max(ann_vec.shape))
        cat_lbl = [x.label for x in patch.annotations if x.label.id < 102][0]
        print '#%d is a %s' % (p_idx, cat_lbl.name)
        cat_tuples.append((cat_lbl.parent_id, cat_lbl.id))
        cat_set.add(cat_lbl.parent_id)
        cat_set.add(cat_lbl.id)

    cat_ids = sorted(cat_set)
    cats = np.zeros((len(patch_ids), len(cat_ids)))
    for idx, item in enumerate(cat_tuples):
        cats[idx, cat_ids.index(item[0])] = 1
        cats[idx, cat_ids.index(item[1])] = 1

    labels = {}
    labels['attributes'] = attrs
    labels['category'] = cats
    labels['attr_ids'] = label_ids
    labels['cat_ids'] = cat_ids
    labels['patch_ids'] = patch_ids
    return labels

# find all patches with labels X that don't have labels Y
# use sql intersect (read about sqlalchemy intersect)
def find_patches(with_lbls, without_lbls, or_lbls, area=config.min_patch_area, splittype=[]):
    '''
    returns list of patch_ids that have annotations with <with_lbls> label ids
    but do not have annotations with <without_lbls> label ids
    '''
    without_lbls_str = ', '.join([str(lbl) for lbl in without_lbls])
    without_stmt = ''
    if without_lbls_str != '':
        without_stmt = ('select id from patch where id not in '
                        '(select distinct a.patch_id from annotation a where a.label_id in (%s)) '
                        % (without_lbls_str))

    subqueries = []
    for lbl_id in with_lbls:
        subqueries.append( 'select distinct a.patch_id from annotation a where a.label_id = %d' % (lbl_id))
    with_stmt = ' intersect '.join(subqueries)

    subqueries = []
    for lbl_id in or_lbls:
        subqueries.append( 'select distinct a.patch_id from annotation a where a.label_id = %d' % (lbl_id))
    or_stmt = ' union '.join(subqueries)

    stmt = ''
    if without_stmt != '':
        stmt += '(%s)' % without_stmt
    if with_stmt != '':
        if stmt != '':
            stmt += ' intersect '
        stmt += '(%s)' % with_stmt
    if or_stmt != '':
        if stmt != '':
            stmt += ' intersect '
        stmt += '(%s)' % or_stmt

    # checking appropriate patch size
    stmt += ' intersect select id from patch where area >= %d ' % area

    # checking patch is from correct train/val/test split
    if splittype != []:
        stmt += " intersect select p.id from image im, patch p where p.image_id = im.id and im.type = '%s' " % splittype
    
    db_res = db.engine.execute(stmt).fetchall()    
    patch_ids = [x[0] for x in db_res]
    return patch_ids

def find_images(with_lbls, without_lbls, or_lbls):
    '''
    returns list of patch_ids that have annotations with <with_lbls> label ids
    but do not have annotations with <without_lbls> label ids
    '''
    without_lbls_str = ', '.join([str(lbl) for lbl in without_lbls])
    without_stmt = ''
    if without_lbls_str != '':
        without_stmt = ('select id from image where id not in '
                        '(select distinct a.image_id from annotation a where a.label_id in (%s)) '
                        % (without_lbls_str))

    subqueries = []
    for lbl_id in with_lbls:
        subqueries.append( 'select distinct a.image_id from annotation a where a.label_id = %d' % (lbl_id))
    with_stmt = ' intersect '.join(subqueries)

    subqueries = []
    for lbl_id in or_lbls:
        subqueries.append( 'select distinct a.image_id from annotation a where a.label_id = %d' % (lbl_id))
    or_stmt = ' union '.join(subqueries)

    stmt = ''
    if without_stmt != '':
        stmt += '(%s)' % without_stmt
    if with_stmt != '':
        if stmt != '':
            stmt += ' intersect '
        stmt += '(%s)' % with_stmt
    if or_stmt != '':
        if stmt != '':
            stmt += ' intersect '
        stmt += '(%s)' % or_stmt
    db_res = db.engine.execute(stmt).fetchall()      
    img_ids = [x[0] for x in db_res]
    return img_ids


def find_positives(pos_lbl_ids, neg_lbl_ids, cat_id, splittype=[]):
    '''
    returns list of patch_ids that have positive annotations with <pos_lbl_ids> label ids
    but have negative annotations with <neg_lbl_ids> label ids
    '''
    attr_parent_id = 407
    subqueries = []
    count = 2
    for lbl_id in pos_lbl_ids:
        # check if we are looking for object labels or attr labels
        if lbl_id < attr_parent_id:
            subqueries.append( ('select a.patch_id '
                                'from annotation a, annotation_vec_mat_view avm '
                                'where a.value is True '
                                'and a.patch_id = avm.patch_id '
                                'and a.label_id = %d ' % (lbl_id)))
        else:
            subqueries.append( ('select patch_id from '
                                '(select count(*), a.patch_id '
                                'from annotation a where a.value is True '
                                'and a.label_id = %d group by a.patch_id) as cnt '
                                'where count >= %d' % (lbl_id, count)))

    for lbl_id in neg_lbl_ids:
        # check if we are looking for object labels or attr labels
        if lbl_id < attr_parent_id:
            subqueries.append( ('select a.patch_id '
                                'from annotation a, annotation_vec_mat_view avm '
                                'where a.patch_id = avm.patch_id '
                                'and not a.label_id = %d ' % (lbl_id)))
        else:
            subqueries.append( ('select patch_id from '
                                '(select count(*), a.patch_id '
                                'from annotation a where a.value is False '
                                'and a.label_id = %d group by a.patch_id) as cnt '
                                'where count >= %d' % (lbl_id, count)))
    if cat_id != []:
        subqueries.append('select patch_id from annotation where value is True and label_id = %d' % cat_id)
    if splittype != []:
        subqueries.append("select p.id from image im, patch p where p.image_id = im.id and im.type = '%s' " % splittype)
    stmt = ' intersect '.join(subqueries)

    db_res = db.engine.execute(stmt).fetchall()
    patch_ids = [x[0] for x in db_res]
    return patch_ids


def get_num_labels(category = 0):
    stmt = "select count(a.patch_id), sum(count) from (select a.patch_id, count(distinct a.label_id) as count from annotation a, hit_response h, worker w, label lbl where a.label_id =lbl.id and lbl.parent_id = 407 and a.hit_id = h.id and h.worker_id = w.id and (w.is_blocked = false) group by a.patch_id) as tmp"
    if category > 0:
        stmt += ", annotation a where a.patch_id = tmp.patch_id and a.label_id = %d" % category
    print 'Category: %s' % Label.query.get(category).name
    num_patch, num_lbl = db.engine.execute(stmt).fetchall()[0]
    num_obj_attr_labels = int(num_lbl)
    num_unique_patches = int(num_patch)
    print '%d (%.2f M) total labels' % (num_obj_attr_labels, num_obj_attr_labels/1000000.0)
    print '%d (%.2f k) unique object instances' % (num_unique_patches, num_unique_patches/1000.0)
    return num_unique_patches, num_obj_attr_labels

def get_instances_by_num_labels(lower_q, upper_q, categories):
    '''
    find instances with >= lower_q but <= upper_q
    returns list of patch_ids and number of distict labels generated for that patch
    does not ensure that each label has repeated turker annotations
    '''
    stmt = "select tmp.patch_id, tmp.count from (select a.patch_id, count(distinct a.label_id) as count from annotation a, hit_response h, worker w, label lbl where a.label_id =lbl.id and lbl.parent_id = 407 and a.hit_id = h.id and h.worker_id = w.id and w.is_blocked is false group by a.patch_id) as tmp, annotation a where a.patch_id = tmp.patch_id and a.label_id in (%s) and tmp.count >= %d and tmp.count <= %d" % (cat_str, lower_q, upper_q)
    instances = db.engine.execute(stmt).fetchall()
    
    return instances
