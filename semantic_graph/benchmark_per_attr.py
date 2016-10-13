#!/usr/bin/env python
import os, sys
import argparse
import datetime

from sklearn.externals import joblib
import numpy as np

import cooccurrence

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--mail_addr", type=str, help='email address to send updates to')
    # parser.add_argument("-w", "--wait_sec", type=int, help='time between email updates')
    # parser.add_argument("-c", "--consensus", action="store_true", help='calc consensus agreement')
    # parser.add_argument("--worker", type=int, help='worker id to check')
    # args = parser.parse_args()

    # if args.consensus:
    #     consensus_count(args.worker, print_on = True)
    # else:
    #     filter_workers_daemon(args.mail_addr, args.wait_sec)

    # labels = joblib.load('data/sun_attrs.jbl')


    # bin_labels = np.zeros(labels['labels_cv'].shape)
    # for r in range(labels['labels_cv'].shape[0]):
    #     for c in range(labels['labels_cv'].shape[1]):
    #         bin_labels[r,c] = 1 if labels['labels_cv'][r,c] > 0.5 else 0
    # bin_labels_train = bin_labels[:7000][:]
    # bin_labels_test = bin_labels[7000:8000][:]

    # bin_labels = joblib.load('data/person_exhaustive_labeled_set.jbl')#scenes_exhaustive_mat.jbl')
    # print bin_labels.keys()
    # # bin_labels = bin_labels['attributes']
    # bin_labels = np.hstack((bin_labels['category'], bin_labels['attributes']))


    # TODO:
    # load labeled sets from each group
    all_attrs = joblib.load('data/object_attr_sublist.jbl')
    lbls = {}
    attrs = {}
    groups = ['person', 'vehicle', 'food', 'animal']
    num_ex_train = 0
    num_ex_test = 0
    num_cat = 0
    for g in groups:
        lbls[g] = joblib.load('data/'+g+'_exhaustive_labeled_set.jbl')
        num_ex_train += int(0.95*lbls[g]['attributes'].shape[0])
        num_ex_test += lbls[g]['attributes'].shape[0] - int(0.95*lbls[g]['attributes'].shape[0])
        attrs[g] = joblib.load('data/'+g+'_attr_sublist.jbl')
        num_cat += lbls[g]['category'].shape[1]
    all_attr_lbls = {}
    all_attr_lbls['train'] = np.zeros((num_ex_train,len(all_attrs)))
    all_attr_lbls['test'] = np.zeros((num_ex_test,len(all_attrs)))
    all_cat_lbls = {}
    all_cat_lbls['train'] = np.zeros((num_ex_train,num_cat))
    all_cat_lbls['test'] = np.zeros((num_ex_test,num_cat))
    cat_subind = {}
    cat_subind['person'] = [num_cat - 1]
    cat_subind['food'] = range(0, lbls['food']['category'][0].shape[0])+[num_cat - 2]
    cat_subind['animal'] = range(lbls['food']['category'][0].shape[0], 
                                 lbls['food']['category'][0].shape[0]+lbls['animal']['category'][0].shape[0])+[num_cat - 3]
    cat_subind['vehicle'] = range(lbls['food']['category'][0].shape[0]+lbls['animal']['category'][0].shape[0], 
                                 lbls['food']['category'][0].shape[0]+lbls['animal']['category'][0].shape[0]+lbls['vehicle']['category'][0].shape[0])+[num_cat - 4]
    aidx = 0
    tidx = 0
    for g in groups:
        attr_subind = [all_attrs.index(x) for x in attrs[g]]
        cs = cat_subind[g]
        for idx in range(lbls[g]['category'].shape[0]):
            if idx < int(0.95*lbls[g]['attributes'].shape[0]):
                all_attr_lbls['train'][aidx][attr_subind] = lbls[g]['attributes'][idx]
                all_cat_lbls['train'][aidx][cat_subind[g]] = lbls[g]['category'][idx]
                aidx += 1
            else:
                all_attr_lbls['test'][tidx][attr_subind] = lbls[g]['attributes'][idx]
                all_cat_lbls['test'][tidx][cat_subind[g]] = lbls[g]['category'][idx]
                tidx += 1
        
    # concat with category encoding and without
    bin_labels_train = np.hstack((all_cat_lbls['train'],all_attr_lbls['train']))
    bin_labels_test = np.hstack((all_cat_lbls['test'],all_attr_lbls['test']))


    sys.exit()

    res = {}
    res['gt'] = bin_labels_test
    res['test'] = {}
    ela_types = {'rand', 'pop', 'backoff', 'dist'}
    for et in ela_types:
        print '***** %s *****' % et
        res['test'][et] = {}
        dtree = {}
        for numq in [5, 10, 20, 30, 40, 50, 80, 100]:

            sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et, ela_limit_type = 'numq', ela_limit =numq)
            
            res['test'][et][numq] = []

            for ind, row in enumerate(bin_labels_test):
                item = sgraph.test(row, known_inds = range(num_cat))
                res['test'][et][numq].append(item[0])
                print 'numq = %d: rec %d = %.2f' % (numq, ind, item[2])

            dtree = sgraph.dtree

    joblib.dump(res,'data/per_attr_rec_benchmark_mle_et_all_object_hier.jbl')

    # run both
    bin_labels_train = all_attr_lbls['train']
    bin_labels_test = all_attr_lbls['test']

    res = {}
    res['gt'] = bin_labels_test
    res['test'] = {}
    ela_types = {'rand', 'pop', 'backoff', 'dist'}
    for et in ela_types:
        print '***** %s *****' % et
        res['test'][et] = {}
        dtree = {}
        for numq in [5, 10, 20, 30, 40, 50, 80, 100]:

            sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et, ela_limit_type = 'numq', ela_limit =numq)
            
            res['test'][et][numq] = []

            for ind, row in enumerate(bin_labels_test):
                item = sgraph.test(row, known_inds = range(num_cat))
                res['test'][et][numq].append(item[0])
                print 'numq = %d: rec %d = %.2f' % (numq, ind, item[2])

            dtree = sgraph.dtree


    joblib.dump(res,'data/per_attr_rec_benchmark_mle_et_all_object_attr_only.jbl')
