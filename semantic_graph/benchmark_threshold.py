#!/usr/bin/env python
import os
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

    labels = joblib.load('data/sun_attrs.jbl')


    bin_labels = np.zeros(labels['labels_cv'].shape)
    for r in range(labels['labels_cv'].shape[0]):
        for c in range(labels['labels_cv'].shape[1]):
            bin_labels[r,c] = 1 if labels['labels_cv'][r,c] > 0.5 else 0
    bin_labels_train = bin_labels[:7000][:]
    bin_labels_test = bin_labels[7000:8000][:]


    # res = {}
    # ela_types = ['rand', 'pop', 'dist'] # 'backoff', 
    res = joblib.load('data/sun_attr_rec_benchmark_mle_et_threshold.jbl')
    ela_types = ['backoff']
    for et in ela_types:
        print '***** %s *****' % et
        res[et] = {}
        for thresh in np.arange(0.005, 0.055, 0.005):
            dtree = {}
            print 'threshold : '+str(thresh)
            sgraph = cooccurrence.SGraph(train=bin_labels_train, dtree=dtree, ela_type=et, ela_limit_type = 'threshold', ela_limit=thresh)
            
            res[et][thresh] = {}
            res[et][thresh]['rec'] = []
            res[et][thresh]['numq'] = []

            for ind, row in enumerate(bin_labels_test[:100][:]):
                item = sgraph.test(row)
                res[et][thresh]['rec'].append(item[2])
                res[et][thresh]['numq'].append(len(item[3]))
                print 'thresh = %.3f: rec %d = %.2f, numq = %d' % (thresh, ind, item[2], len(item[3]))

            dtree = sgraph.dtree

    joblib.dump(res,'data/sun_attr_rec_benchmark_mle_et_threshold_wo_dtree.jbl')
