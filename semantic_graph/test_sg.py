import os
import time
import random

from sklearn.externals import joblib
import numpy as np

import config
import cooccurrence
import entropy

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def test_train(labels):
    return labels[:labels.shape[0]/2,:], labels[labels.shape[0]/2:,:]

def test_sg(sgraph, test):

    try:
        with Timer() as t:
            prec = 0
            rec = 0
            rinds = [random.randint(0,test.shape[0]-1) for r in range(10)]
            all_idx = []
            for row, query in enumerate(test[rinds,:]):
                est, lidx, p, r = sgraph.test(query)
                prec += p
                rec += r
                all_idx += lidx
            print [config.attr[a] for a in set(all_idx)]
            prec = prec/float(row+1)
            rec = rec/float(row+1)
    finally:
        print '_______________________'
        print 'row %d' % row
        print query
        print('precision = %f\n recall = %f' % (prec, rec))
        print('testing took %.03f sec. %.03f mins.' % (t.interval, t.interval/60.0))
    return prec, rec

def compare_cooc():
    test, train = test_train(config.binlabels)
    avg_prec = []
    avg_rec = []
    num_q = range(5,105,5)+[102]
    for num in num_q:
        dtree = joblib.load(os.path.join(config.home_dir, 'semantic_graph/cooc_tree.jbl'))
        sgraph = cooccurrence.SGraph(train,num, dtree)
        p, r = test_sg(sgraph, test)
        joblib.dump(dtree, os.path.join(config.home_dir, 'semantic_graph/cooc_tree.jbl'))
        avg_prec.append(p)
        avg_rec.append(r)
    print avg_prec
    print avg_rec
    res = { 'num_questions': num_q, 'rec': avg_rec, 'prec': avg_prec}
    joblib.dump(res, os.path.join(config.home_dir, 'data/cooccurrence_sun_attributes_1000rep.jbl'))

def compare_entropy():
    test, train = test_train(config.binlabels)
    avg_prec = []
    avg_rec = []
    num_q = range(5,105,5)+[102]
    for num in num_q:
        dtree = joblib.load(os.path.join(config.home_dir, 'semantic_graph/entropy_tree.jbl'))
        sgraph = entropy.SGraph(train,num, dtree)
        p, r = test_sg(sgraph, test)
        joblib.dump(dtree, os.path.join(config.home_dir, 'semantic_graph/entropy_tree.jbl'))
        avg_prec.append(p)
        avg_rec.append(r)
    print avg_prec
    print avg_rec
    res = { 'num_questions': num_q, 'rec': avg_rec, 'prec': avg_prec}
    joblib.dump(res, os.path.join(config.home_dir, 'data/entropy_sun_attributes_10rep.jbl'))

# entropy method is always returning same set of questions regardless of answers. test to debug
#test calculated entropy per dim given same train set but different test vectors (some known some unknown dim)
