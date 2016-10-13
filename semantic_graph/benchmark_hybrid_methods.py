#!/usr/bin/env python
import os
import argparse
import datetime

from sklearn.externals import joblib
import numpy as np

import cooccurrence

if __name__ == "__main__":

    # load labeled sets from each group
    all_attrs = joblib.load('data/object_attr_sublist.jbl')
    lbls = {}
    attrs = {}
    groups = ['person']#, 'vehicle', 'food', 'animal']
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
    # cat_subind['food'] = range(0, lbls['food']['category'][0].shape[0])+[num_cat - 2]
    # cat_subind['animal'] = range(lbls['food']['category'][0].shape[0], 
    #                              lbls['food']['category'][0].shape[0]+lbls['animal']['category'][0].shape[0])+[num_cat - 3]
    # cat_subind['vehicle'] = range(lbls['food']['category'][0].shape[0]+lbls['animal']['category'][0].shape[0], 
    #                              lbls['food']['category'][0].shape[0]+lbls['animal']['category'][0].shape[0]+lbls['vehicle']['category'][0].shape[0])+[num_cat - 4]
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
    print 'num train', bin_labels_train.shape[0]
    print 'num test', bin_labels_test.shape[0]
    res = {}

    ela_types = {'rand', 'pop', 'backoff', 'dist'}
    for et in ela_types:
        print '***** %s *****' % et
        res[et] = {}
        dtree = {}
        for numq in [10, 20, 30, 40, 50, 80, 100]:

            sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et, ela_limit_type = 'numq', ela_limit =numq)
            
            res[et][numq] = []

            for ind, row in enumerate(bin_labels_test[:100]):
                item = sgraph.test(row, known_inds = range(num_cat))
                res[et][numq].append(item[2])
                print 'numq = %d: rec %d = %.2f' % (numq, ind, item[2])

            dtree = sgraph.dtree


    # test hybrid methods, use simple method for first 10 attrs, then more complicated methods for the rest
    ela_types = {'rand+dist', 'pop+dist', 'rand+backoff', 'pop+backoff'}
    for et in ela_types:
        print '***** %s *****' % et
        res[et] = {}
        dtree = {}
        # add starting point at 10
        numq = 10
        et1, et2 = et.split('+')
        sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et1, ela_limit_type = 'numq', ela_limit =numq)
        
        res[et][numq] = []

        for ind, row in enumerate(bin_labels_test[:100]):
            item = sgraph.test(row, known_inds = range(num_cat))
            res[et][numq].append(item[2])
            print 'numq = %d: rec %d = %.2f' % (numq, ind, item[2])

        dtree = sgraph.dtree
        # continue with secondary method
        for numq in [20, 30, 40, 50, 80, 100]:

            sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et2, ela_limit_type = 'numq', ela_limit =numq)
            
            res[et][numq] = []

            for ind, row in enumerate(bin_labels_test):
                item = sgraph.test(row, known_inds = range(num_cat))
                res[et][numq].append(item[2])
                print 'numq = %d: rec %d = %.2f' % (numq, ind, item[2])

            dtree = sgraph.dtree



    joblib.dump(res,'data/attr_rec_benchmark_mle_et_person_hybrid.jbl')
    sys.exit()


    # run with only attribute info, no categories
    bin_labels_train = all_attr_lbls['train']
    bin_labels_test = all_attr_lbls['test']


    res = {}

    ela_types = {'rand', 'pop', 'backoff', 'dist'}
    for et in ela_types:
        print '***** %s *****' % et
        res[et] = {}
        dtree = {}
        for numq in [10, 20, 30, 40, 50, 80, 100]:

            sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et, ela_limit_type = 'numq', ela_limit =numq)
            
            res[et][numq] = []

            for ind, row in enumerate(bin_labels_test):
                item = sgraph.test(row)
                res[et][numq].append(item[2])
                print 'numq = %d: rec %d = %.2f' % (numq, ind, item[2])

            dtree = sgraph.dtree


    # test hybrid methods, use simple method for first 10 attrs, then more complicated methods for the rest
    ela_types = {'rand+dist', 'pop+dist', 'rand+backoff', 'pop+backoff'}
    for et in ela_types:
        print '***** %s *****' % et
        res[et] = {}
        dtree = {}
        # add starting point at 10
        numq = 10
        et1, et2 = et.split('+')
        sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et1, ela_limit_type = 'numq', ela_limit =numq)
        
        res[et][numq] = []

        for ind, row in enumerate(bin_labels_test):
            item = sgraph.test(row, known_inds = range(num_cat))
            res[et][numq].append(item[2])
            print 'numq = %d: rec %d = %.2f' % (numq, ind, item[2])

        dtree = sgraph.dtree
        # continue with secondary method
        for numq in [20, 30, 40, 50, 80, 100]:

            sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et2, ela_limit_type = 'numq', ela_limit =numq)
            
            res[et][numq] = []

            for ind, row in enumerate(bin_labels_test):
                item = sgraph.test(row)
                res[et][numq].append(item[2])
                print 'numq = %d: rec %d = %.2f' % (numq, ind, item[2])

            dtree = sgraph.dtree

    joblib.dump(res,'data/attr_rec_benchmark_mle_et_all_object_attr_only_hybrid.jbl')
