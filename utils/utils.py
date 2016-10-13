#!/usr/bin/env python
import errno
import json
import math
import os
import random
import sys
import datetime
import subprocess

import numpy as np

from app import db
from app.models import *
import config
import jfif

'''

Frequently used small functions

'''
def get_first_db_res(stmt):
    try:
        return db.engine.execute(stmt).fetchall()[0][0]
    except IndexError, e:
        return None

def get_all_db_res(stmt):
    return db.engine.execute(stmt).fetchall()

def time_stamp(print_str):
    ts = str(datetime.datetime.now())
    print '%s -- %s' % (ts, print_str)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

# def check_corrupt(feat_file):
#     '''
#     Check if file can be opened. True = corrupt file False = not corrupt
#     If file does not exist returns False (not corrupt).
#     '''

#     iscorrupt = False
#     if not os.path.exists(feat_file):
#         print 'File does not exist '+feat_file
#         return iscorrupt

#     try:
#         curf = joblib.load(feat_file)
#     except (IOError, EOFError) as e:
#         print 'Can''t open file '+feat_file
#         iscorrupt = True

#     return iscorrupt


def get_imgs_from_dir(img_dir):
    '''
    Returns list of full paths for all images in input dir
    '''
    img_files = []

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(config.img_file_exts):
                img_files.append(os.path.join(root, file))

    return img_files

def get_feats_from_dir(feat_dir):
    '''
    Returns list of full paths for all feature files in input dir
    '''
    feat_files = []

    for root, dirs, files in os.walk(feat_dir):
        for file in files:
            if file.endswith('.jbl'):
                feat_files.append(os.path.join(root, file))

    return feat_files

def email_notify(message, subject, to_addr):
    output = subprocess.check_output("echo '%s' | mail -s '%s' %s" % (message, subject, to_addr), shell=True)
    return output

# def save_file_to_s3(fname, local_file, bucket_name):
#     s3conn = s3.S3Connection()

#     # check that bucket exists, if not make bucket.
#     print_ls = False
#     buckets = s3conn.ls(print_ls)

#     if not bucket_name in buckets:
#         print 'Making new bucket %s' % bucket_name
#         s3conn.mkdir(bucket_name)
#     print 'Saving %s to S3' % (local_file)
#     s3conn.put(bucket_name, fname, local_file)

# def load_feature_file(fname, feat_dir, is_bucket_name=False):

#     if not is_bucket_name:
#         feat_fname = os.path.join(feat_dir, fname+'.jbl')
#         im_feat = joblib.load(feat_fname)
#     else:
#         return 'TODO: s3 option not implemented'
#         # bucket_name = feat_dir
#         # s3conn = s3.S3Connection()

#         # # check that bucket exists
#         # print_ls = False
#         # buckets = s3conn.ls(print_ls)

#         if not bucket_name in buckets:
#             print 'Bucket %s does not exist' % bucket_name
#             return []

#         file_to_write = cStringIO.StringIO()
#         # s3conn.get(bucket_name, fname, file_to_write)
#         im_feat = file_to_write.getvalue()

#     return im_feat

# def get_feat_save_path(savedir, fname):
#     if config.split_save_path:
#         last_two_chars = fname[-2:]
#         tmpdir = os.path.join(savedir,last_two_chars)
#         if not os.path.exists(tmpdir):
#             os.mkdir(tmpdir)
#     else:
#         tmpdir = savedir
#     return tmpdir

def slugify(word):
    word = word.replace("[", "")
    word = word.replace("]", "")
    word = word.replace("'", "")
    word = word.replace(",", "")
    word = word.replace(" ", "-")
    word = word.replace("/", "_")
    return word

def load_img_dir(img_dir, symlink_prefix, url_prefix):
    time_stamp('start')
    img_files = get_imgs_from_dir(img_dir)
    time_stamp('images to load = %d' % (len(img_files)))
    
    # In sets of 1000, add images in list
    step = 1000
    add_imgs(img_files, symlink_prefix, url_prefix, step)

def add_img(img_file, symlink_prefix, url_prefix):
    if Image.query.filter_by(location = img_file).first() == None:
        ext = os.path.splitext(img_file)[1]
        try:
            mime = config.mime_dictionary[ext]
        except KeyError, e:
            mime = ''
        if ext == '.jpg':
            latitude, longitude = jfif.gps(img_file)
        else:
            latitude, longitude = (None, None)

        url = img_file.replace(symlink_prefix, '')
        url = os.path.join(url_prefix, url)

        blob = Image(location = img_file, ext = ext, mime = mime,
                           url = url, latitude = latitude, longitude = longitude)
        db.session.add(blob)
        db.session.commit()
        
def add_imgs(img_files, symlink_prefix, url_prefix, step):
    stmt = 'insert into image values '
    cnt = 0
    start_cnt = 0
    stop_cnt = 0
    for img_file in img_files:
        location = img_file
        ext = os.path.splitext(img_file)[1]
        try:
            mime = config.mime_dictionary[ext]
        except KeyError, e:
            mime = ''
        if ext == '.jpg':
            latitude, longitude = jfif.gps(img_file)
        else:
            latitude, longitude = (None, None)

        url = img_file.replace(symlink_prefix, '')
        url = os.path.join(url_prefix, url)
            
        if cnt > 0:
            stmt += ', '
        stmt += "(default, '%s', '%s', '%s', '%s', %f, %f)" % (ext, mime, location, url, latitude, longitude)
        cnt += 1
        stop_cnt += 1
        
        if cnt > step:
            time_stamp('Adding files idx %d - %d...' % (start_cnt, stop_cnt))
            results = db.engine.execute(stmt)
            stmt = 'insert into image values '
            cnt = 0
            start_cnt = stop_cnt
                       
    if cnt > 0:
        time_stamp('Adding files idx %d - %d...' % (start_cnt, stop_cnt))
        results = db.engine.execute(stmt)

def email_notify(message, subject, to_addr):
    output = subprocess.check_output("echo '%s' | mail -s '%s' %s" % (message, subject, to_addr), shell=True)
    return output
