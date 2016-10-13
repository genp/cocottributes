#!/usr/bin/env python
import os
from itertools import chain
import urllib
import glob
import argparse
import json

from nltk import word_tokenize, Text, pos_tag, ConcordanceIndex
import numpy as np
import wikipedia

def get_text(fname, isurl=False):
    '''
    This is intended to read in a plain text file
    and return an nltk Text object.
    '''

    if not isurl:
        raw = ''
        with open(fname, 'r') as f:
            for x in f:
                raw += x.split('.xml: ', 1)[1]

    else:
        raw = urllib.urlopen(fname).read()

    return tokenize_text(raw.decode('latin-1')) # encoding for nyt text

def tokenize_text(raw):
    tokens = word_tokenize(raw) 
    text = Text(tokens)
    return text

def word_count(word_list):
    '''
    returns list of tuples [(unique word, count)...]
    '''
    uni = list(set(word_list))
    uni_cnt = {}
    for w in uni:
        uni_cnt[w] = word_list.count(w)
    cnts = sorted(zip(uni_cnt.keys(), uni_cnt.values()), key=lambda x: x[1], reverse=True)    
    return cnts

def count_dict(word_dict, corpus):
    for w in word_dict.keys():
        word_dict[w] += corpus.count(w)
    return word_dict

def init_word_dict(words):
    wdict = {}
    for w in words:
        wdict[w] = 0
    print len(wdict.keys())
    return wdict

def table_print(things, table_name=None, limit=100):
    if table_name:
        print table_name
    print '----------------------'
    for idx, item in enumerate(things):
        print '%s %d' % item
        if idx > limit:
            break


def get_attributes_from_wikipedia():
    with open('/Users/gen/coco/objects.txt') as f:
        objects = f.read().splitlines()
        
    # adj, adverb, verb
    parts_of_speach = ['JJ', 'RB', 'VB']

    related_words = {}
    for target_word in objects:
        print target_word 
        related_words[target_word] = {}
        tmp = wikipedia.page(target_word)
        tokens = word_tokenize(tmp.content)
        text = Text(tokens)
        pos = pos_tag(text)
        
        for part in parts_of_speach:
            part_words = [w[0] for w in pos if w[1] == part]
            word_cnt = word_count(part_words)
            words = list(set(part_words))

            word_cc = init_word_dict(words)

            # get concordant words for object

            margin = 50
            cc =  ConcordanceIndex(text.tokens, key = lambda s: s.lower())
            concordance_txt = ([text.tokens[map(lambda x: x-5 if (x-margin) > 0 else 0, [offset])[0]:offset+margin] for offset in cc.offsets(target_word)])
            cc_plain = list(chain(*concordance_txt))
            count_dict(word_cc, cc_plain)
            related_words[target_word][part] = word_cc

    return related_words


def get_attributes_from_coco_captions(obj_ind):
    data_dir = '/data/hays_lab/COCO/coco/'
    with open(os.path.join(data_dir, 'objects.txt')) as f:
        objects = f.read().splitlines()
    obj = objects[obj_ind]
    # adj, adverb, verb
    parts_of_speach = ['JJ', 'RB', 'VB']

    cap_fnames = [os.path.join(data_dir, 'annotations/captions_train2014.json'), os.path.join(data_dir, 'annotations/captions_val2014.json')]

    cap_text = ''

    for fname in cap_fnames:
        with open(fname, 'r') as f:
                cap = json.load(f)
        for item in cap['annotations']:
            line = str(item['caption'])
            if obj in line:
                cap_text += ' '+line

    text = tokenize_text(cap_text)
    pos = pos_tag(text)
    adjs = [w[0] for w in pos if w[1] == 'JJ']
    advs = [w[0] for w in pos if w[1] == 'RB']
    verbs = [w[0] for w in pos if w[1] == 'VB']
    adjs_cnt = word_count(adjs)
    advs_cnt = word_count(advs)
    verbs_cnt = word_count(verbs)
    adjs = list(set(adjs))
    print '********************************************'
    print obj
    print '--------------------------------------------'
    print 'top adjs: '+obj
    table_print(adjs_cnt[:50])
    advs = list(set(advs))
    print '--------------------------------------------'
    print 'top advs: '+obj
    table_print(advs_cnt[:50])
    verbs = list(set(verbs))
    print '--------------------------------------------'
    print 'top verbs: '+obj
    table_print(verbs_cnt[:50])

    # adjs_cc = init_word_dict(adjs)
    # print len(adjs_cc.keys())
    # advs_cc = init_word_dict(advs)
    # verbs_cc = init_word_dict(verbs)
    cc = {}
    cc[obj] = {}
    cc[obj]['adj'] = adjs_cnt
    cc[obj]['adv'] = advs_cnt
    cc[obj]['verb'] = verbs_cnt
    with open('/home/gen/coco_attributes/data/captions_'+obj+'_word_cnt.json', 'w') as out:
        json.dump(cc, out)



if __name__ == "__main__":
    # get sorted list of adjs, advs 
    # loop through a bunch of books
    # books = ['https://www.gutenberg.org/cache/epub/1342/pg1342.txt', 'https://www.gutenberg.org/cache/epub/1400/pg1400.txt']
    # books = glob.glob('/home/gen/nyt-text/*.txt')
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("fname", help="file to parse")
    args = parser.parse_args()
    get_attributes_from_coco_captions(int(args.fname)) # this should be an integer between 0-90
    """
    books = [args.fname]
    # adjs_cc = {}
    # advs_cc = {}
    # verbs_cc = {}

    for fname in books:
        obj = os.path.basename(fname)[:-4]
        text = get_text(fname)
        pos = pos_tag(text)
        adjs = [w[0] for w in pos if w[1] == 'JJ']
        advs = [w[0] for w in pos if w[1] == 'RB']
        verbs = [w[0] for w in pos if w[1] == 'VB']
        adjs_cnt = word_count(adjs)
        advs_cnt = word_count(advs)
        verbs_cnt = word_count(verbs)
        adjs = list(set(adjs))
        print '********************************************'
        print fname
        print '--------------------------------------------'
        print 'top adjs: '+fname
        table_print(adjs_cnt[:50])
        advs = list(set(advs))
        print '--------------------------------------------'
        print 'top advs: '+fname
        table_print(advs_cnt[:50])
        verbs = list(set(verbs))
        print '--------------------------------------------'
        print 'top verbs: '+fname
        table_print(verbs_cnt[:50])

        # adjs_cc = init_word_dict(adjs)
        # print len(adjs_cc.keys())
        # advs_cc = init_word_dict(advs)
        # verbs_cc = init_word_dict(verbs)
        cc = {}
        cc[obj] = {}
        cc[obj]['adj'] = adjs_cnt
        cc[obj]['adv'] = advs_cnt
        cc[obj]['verb'] = verbs_cnt
        with open('/home/gen/nyt-text/'+obj+'_word_cnt.json', 'w') as out:
            json.dump(cc, out)
        """
    #     # get concordant adj, adv for COCO objects
    #     with open('/Users/gen/coco/objects.txt') as f:
    #         objects = f.read().splitlines()

    #     for target_word in objects:
    #         margin = 50
    #         cc =  ConcordanceIndex(text.tokens, key = lambda s: s.lower())
    #         concordance_txt = ([text.tokens[map(lambda x: x-5 if (x-margin) > 0 else 0, [offset])[0]:offset+margin] for offset in cc.offsets(target_word)])
    #         cc_plain = list(chain(*concordance_txt))
    #         count_dict(adjs_cc, cc_plain)
    #         count_dict(advs_cc, cc_plain)
    #         count_dict(verbs_cc, cc_plain)
    # adjs_cc_cnts = sorted(zip(adjs_cc.keys(), adjs_cc.values()), key=lambda x: x[1], reverse=True)
    # advs_cc_cnts = sorted(zip(advs_cc.keys(), advs_cc.values()), key=lambda x: x[1], reverse=True)
    # verbs_cc_cnts = sorted(zip(verbs_cc.keys(), verbs_cc.values()), key=lambda x: x[1], reverse=True)
    # print '--------------------------------------------'
    # print 'top concordant adjs'
    # table_print(adjs_cc_cnts[:50])
    # print '--------------------------------------------'
    # print 'top concordant advs'
    # table_print(advs_cc_cnts[:50])
    # print '--------------------------------------------'
    # print 'top concordant verbs'
    # table_print(verbs_cc_cnts[:50])
