#!/usr/bin/env python
from collections import Counter
import argparse
import json
import os

import numpy as np

from app import db
from app.models import *
from utils.utils import *
from word_extractor import table_print

def words_pop(fname):
    '''
    collect all unique words from each worker for each object
    list in table in descending order of popularity
    ''' 
    collected_attrs = {}
    parent_ids = [x[0] for x in get_all_db_res("select id from label") if x[0] < 102]

    for parent_id in parent_ids:
        stmt = "select distinct (name, worker_id) from word w, hit_response h where w.hit_id = h.id and parent_id = %d" % parent_id
        res = [x[0][1:-1].split(',') for x in get_all_db_res(stmt)]
        obj_name = Label.query.get(parent_id).name
        if len(res) == 0:
            print "%s submitted attributes (total %d):\n" % (obj_name, 0)
            continue
        words, workers = zip(*res)
        word_count = Counter(words)
        collected_attrs[obj_name] = word_count
        table_print(sorted(word_count.items(), key=lambda w:w[1], reverse=True), "%s submitted attributes (total %d):" % (Label.query.get(parent_id).name, len(word_count)), 200)
        print ''
    with open(fname, 'w') as outfile:
        json.dump(collected_attrs, outfile)

def words_pop_hierarchy(fname):
    '''
    collect all unique words from each worker overall, then for lvl 1 of hierarchy, 
    then top 5 unique words per obj not  in hierarchy
    list in table in descending order of popularity
    ''' 
    collected_attrs = {}
    obj_ids = [x[0] for x in get_all_db_res("select id from label") if x[0] < 102]
    lvl_1_ids = [x[0] for x in get_all_db_res("select id from label where id in (select distinct parent_id from label where parent_id < 102) order by id")]

    # most popular attributes overall
    stmt = "select distinct (name, worker_id) from word w, hit_response h where w.hit_id = h.id"
    res = [x[0][1:-1].split(',') for x in get_all_db_res(stmt)]
    words, workers = zip(*res)
    word_count = Counter(words)
    collected_attrs['all'] = word_count
    table_print(sorted(word_count.items(), key=lambda w:w[1], reverse=True), "%s submitted attributes (total %d):" % ('All', len(word_count)), 500)
    print ''

    # lvl 1 hierarchy attribtues
    for lvl_1_obj_id in lvl_1_ids:
        stmt = "select distinct (w.name, worker_id) from word w, hit_response h, label lbl where w.hit_id = h.id and w.parent_id = lbl.id and lbl.parent_id = %d" % lvl_1_obj_id
        res = [x[0][1:-1].split(',') for x in get_all_db_res(stmt)]
        obj_name = Label.query.get(lvl_1_obj_id).name
        if len(res) == 0:
            print "%s submitted attributes (total %d):\n" % (obj_name, 0)
            continue
        words, workers = zip(*res)
        word_count = Counter(words)
        collected_attrs[obj_name] = word_count
        members = Label.query.get(lvl_1_obj_id).name +  ' ('
        members += ', '.join([x[0] for x in get_all_db_res("select name from label where parent_id = %d" % lvl_1_obj_id)]) + ')'
        
        table_print(sorted(word_count.items(), key=lambda w:w[1], reverse=True), "%s submitted attributes (total %d):" % (members, len(word_count)), 200)
        print ''
    with open(fname, 'w') as outfile:
        json.dump(collected_attrs, outfile)

    # # individual objects
    # parent_ids = [x[0] for x in get_all_db_res("select id from label") if x[0] < 102]

    # for parent_id in parent_ids:
    #     stmt = "select distinct (name, worker_id) from word w, hit_response h where w.hit_id = h.id and parent_id = %d" % parent_id
    #     res = [x[0][1:-1].split(',') for x in get_all_db_res(stmt)]
    #     obj_name = Label.query.get(parent_id).name
    #     if len(res) == 0:
    #         print "%s submitted attributes (total %d):\n" % (obj_name, 0)
    #         continue
    #     words, workers = zip(*res)
    #     # check only for words not in parent entry
    #     try:
    #         obj_lvl_1_name = Label.query.get(Label.query.get(parent_id).parent_id).name
    #     except TypeError, e: # there is no lvl 1 id for this item, it is a lvl 1 item
    #         continue
    #     lvl_1_words = collected_attrs[obj_lvl_1_name].keys()
    #     words = list(set(words)-set(lvl_1_words))        
    #     word_count = Counter(words)
    #     collected_attrs[obj_name] = word_count
    #     table_print(sorted(word_count.items(), key=lambda w:w[1], reverse=True), "%s submitted attributes (total %d):" % (Label.query.get(parent_id).name, len(word_count)), 10)
    #     print ''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count unique attribute submissions from MTurkers.')
    parser.add_argument('fname', help='filename (*.json) to save dict of attribute word counts')
    args = parser.parse_args()
    words_pop_hierarchy(args.fname)

