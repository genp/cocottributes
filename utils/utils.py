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
import tqdm
from sklearn.externals import joblib

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


def convert(train_annotations, val_annotations, attributes):
    """
    Convert the original dataset format to an easier one, aka my implementation as specified in DATA.md.
    :param train_annotations: The training annotations dataset from MS COCO which gives us the object bounding boxes and categories.
    :param val_annotations: The validation annotations dataset from MS COCO which gives us the object bounding boxes and categories.
    :param attributes: The MS COCO attributes dataset.
    :return: True if conversion is successful.
    """
    new_scheme = dict()
    new_scheme['ann_attrs'] = {}

    # Create a copy of the attributes
    attrs = list(attributes['attributes'])
    # Sort the keys on their IDs
    attrs.sort(key=lambda x: x['id'])

    new_scheme['attributes'] = attrs

    for i, idx in enumerate(attributes['ann_vecs'].keys()):
        print(i)

        split = attributes['split'][idx]

        # Get the attributes and the corresponding annotation ID
        object_attrs = attributes['ann_vecs'][idx]
        ann_id = attributes['patch_id_to_ann_id'][idx]

        if 'train' in split:
            annotations = train_annotations
        elif 'val' in split:
            annotations = val_annotations

        object_annotation = [a for a in annotations if a['id'] == ann_id][0]

        try:
            new_scheme['ann_attrs'][object_annotation['id']] = {'attrs_vector': object_attrs, 'split': split}
        except (Exception,):
            print("Error for idx={}".format(idx))
            pass

    joblib.dump(new_scheme, '../../MSCOCO/cocottributes_new_version.jbl')


def get_image_crop(img, x, y, width, height, crop_size=224, padding=16):
    """
    Get the image crop for the object specified in the COCO annotations.
    We crop in such a way that in the final resized image, there is `padding` amount of image data around the object.
    This is the same as is used in RCNN to allow for additional image context.
    :param img: The image ndarray
    :param x: The x coordinate for the start of the bounding box
    :param y: The y coordinate for the start of the bounding box
    :param width: The width of the bounding box
    :param height: The height of the bounding box
    :param crop_size: The final size of the cropped image. Needed to calculate the amount of context padding.
    :param padding: The amount of context padding needed in the image.
    :return:
    """
    # Scale used to compute the new bbox for the image such that there is surrounding context.
    # The way it works is that we find what is the scaling factor between the crop and the crop without the padding
    # (which would be the original tight bounding box).
    # `crop_size` is the size of the crop with context padding.
    # The denominator is the size of the crop if we applied the same transform with the original tight bounding box.
    scale = crop_size / (crop_size - padding * 2)

    # Calculate semi-width and semi-height
    semi_width = width / 2
    semi_height = height / 2

    # Calculate the center of the crop
    centerx = x + semi_width
    centery = y + semi_height

    # We get the crop using the semi- height and width from the center of the crop.
    # The semi- height and width are scaled accordingly.
    # We also ensure the numbers are not negative (Python3 sets the dimension to 0 otherwise which results in an error)
    lowy = int(max(0, round(centery - (semi_height * scale))))
    highy = int(max(0, round(centery + (semi_height * scale))))
    lowx = int(max(0, round(centerx - (semi_width * scale))))
    highx = int(max(0, round(centerx + (semi_width * scale))))

    crop_img = img[lowy:highy, lowx:highx]
    return crop_img


def get_images_list(annotations, attributes, data_root=".", split="train"):
    """
    Helper function to retrieve a list of JSON objects for each image in the dataset
    with all the relevant data which is:
    1. Image path (path).
    2. Split to which the image belongs to (split).
    3. The bounding box for the object if applicable (bbox).
    4. COCO Attributes vector (attrs_vector).
    5. The ID of the annotation in the COCO dataset (id).
    :param annotations: The COCO Annotations dataset.
    :param attributes: The COCO Attributes dataset.
    :param data_root: The path to the directory where the datasets are located on the drive.
    :param split: The data split (train, val, test, etc.) to which the image belongs to.
    :return: A list of JSON objects for each image with all the data specified prior.
    """
    # The final list to be returned
    images_list = []

    # A list of all the images in the split so as to iterate over
    split_img_list = []
    for idx in attributes['ann_attrs'].keys():
        d = attributes['ann_attrs'][idx]
        if split in d['split']:
            d['id'] = idx
            split_img_list.append(d)

    for i, ann in tqdm(enumerate(split_img_list), total=len(split_img_list)):
            # We use the ann id to find the annotation in the annotations dataset.
            # Since ID is unique, there should always be only one.
            object_annotation = [a for a in annotations if a['id'] == ann['id']][0]

            # Get the image ID and the bounding box for the object
            img_id = object_annotation['image_id']

            ann['path'] = osp.join(data_root,
                                   "{0}2014".format(split),
                                   "COCO_{0}2014_{1:012d}.jpg".format(split, img_id))
            ann['bbox'] = object_annotation['bbox']

            # Get the object attributes as an array of 1s and 0s.
            ann['attrs_vector'] = np.array([np.float(x > 0) for x in ann['attrs_vector']])

            images_list.append(ann)

    return images_list
