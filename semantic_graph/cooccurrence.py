import heapq
import random

import numpy as np
import scipy.spatial.distance as scpydist




class SGraph:
    
    def __init__(self, **kwargs):
        '''
        labels: exhaustively labeled training set - binary values only
        dtree: precalculated decision data structure (dict)
        kwargs can contain:
            numq: number of questions that can be asked per test image
            threshold: lower limit for likelihood of next question. 
                       SGraph stops queries after threshold is reached.
        '''
        self.__dict__.update(kwargs)
        # self.labels = kwargs.get('labels')
        # self.dtree = kwargs.get('dtree') if kwargs.get('dtree') else {}
        # self.ela_type = kwargs.get('ela_type') # ela_type = {'rand', 'pop', 'backoff', 'dist'}
        # self.ela_limit_type = kwargs.get('ela_limit_type') # ela_limit_type = {'numq', 'threshold'}
        # self.ela_limit = kwargs.get('ela_limit')
        if self.ela_type == 'pop':
            self.occur_mat_train = occur_mat(self.train)
        else:
            self.occur_mat_train = []

        self._est_funcs = { 'numq': self.est_numq,
                            'threshold': self.est_threshold }
    
    def get_next_query(self, cur_label, prev_label = None):
        if -1 not in cur_label:
            return -1, 0.0, 0

        # check dtree
        knowns = get_knowns(cur_label)
        try:
            check, mle = check_dtree(self.dtree, knowns)
        except TypeError, e:
            check = -1
        
        if not check == -1:
            next_ind = check
            new_match = 0
        else:
            next_ind, mle, new_match = get_next_query_ind(cur_label, self.train, self.occur_mat_train, self.ela_type, prev_label)
            # TODO: decision tree currently disabled, decide whether or not to keep it
            # update_dtree(self.dtree,next_ind,knowns, mle)

        # print 'next_ind, mle, new_match: %d, %.3f, %d' % (next_ind, mle, new_match)
        return next_ind, mle, new_match

    def test(self, true_label, known_inds = None):
        est, query_inds = self._est_funcs[self.ela_limit_type](true_label, known_inds)

        num_tp = sum([elem for ind,elem in enumerate(est) if true_label[ind] == 1])
        pos = sum(est)
        if pos == 0:
            prec = 1.0
            # print 'didnt estimate any true values'
        else:
            prec = num_tp/pos
        true_pos = sum(true_label)
        if true_pos == 0:
            rec = 1.0
        else:
            rec = num_tp/true_pos
        return est, prec, rec, query_inds


    def est_numq(self, true_label, known_inds = None):
        est = -1*np.ones(true_label.shape)
        if known_inds != None:
            est[known_inds] = true_label[known_inds]
        prev_inds = []
        prev_label = -1*np.ones(true_label.shape)
        num_past_steps = 1
        for j in range(self.ela_limit):
            idx, mle, new_match = self.get_next_query(est, prev_label)

            prev_inds.append(idx)
            # print 'prev inds: '+str(prev_inds)

            # TODO: if mle is zero, move on to other approach??
            if new_match and mle >= 0.0001:
                if len(prev_inds) > 1:
                    prev_label[prev_inds[-1-num_past_steps:-1]] = true_label[prev_inds[-1-num_past_steps:-1]]
                    num_past_steps = 1

            else:
                num_past_steps += 1
                # print 'updated past steps ' + str(num_past_steps)

            est[idx] = true_label[idx]
            # print 'value at %d = %d' % (idx, est[idx])
            # print num_past_steps
            # print all(est == prev_label)

        # if attribute has not been labeled, assume false
        est[np.where(est == -1.0)] = 0.0
        return est, prev_inds

    def est_threshold(self, true_label, known_inds = None):
        est = -1*np.ones(true_label.shape)
        if known_inds != None:
            est[known_inds] = true_label[known_inds]
        prev_inds = []
        prev_label = -1*np.ones(true_label.shape)
        num_past_steps = 1
        idx, mle, new_match = self.get_next_query(est, prev_label)
        while mle > self.ela_limit and idx != -1:
            prev_inds.append(idx)

            if new_match:
                if len(prev_inds) > 1:
                    prev_label[prev_inds[-1-num_past_steps:-1]] = true_label[prev_inds[-1-num_past_steps:-1]]
                    num_past_steps = 1
            else:
                num_past_steps += 1
                # print 'updated past steps ' + str(num_past_steps)

            est[idx] = true_label[idx]
            # print 'value at %d = %d' % (idx, est[idx])
            # print num_past_steps

            idx, mle, new_match = self.get_next_query(est, prev_label)

        # if attribute has not been labeled, assume false
        est[np.where(est == -1.0)] = 0.0
        return est, prev_inds

def match_rows(query, train, match_inds):
    match_rows = np.array([row for i,row in enumerate(train) if all(row[match_inds] == query[match_inds])])
    return match_rows

def closest_rows(query, train, match_inds, num_nn):
    # print 'closest rows'
    mesh = np.ix_(range(train.shape[0]), match_inds)
    res = scpydist.cdist([query[match_inds]], train[mesh], 'euclidean')
    # print res.shape
    # print res.max()
    # print res.min()
    closest_inds = heapq.nsmallest(num_nn, range(train.shape[0]), key = lambda x: res[0,x])
    # print 'distances of NN: '+str([res[0,x] for x in closest_inds])
    return train[closest_inds, :]

def cooc_mat(labels):
    '''
    create cooccurence matrix given a set of labeled instances.
    input: labels - m x n matrix, m = number of instances, n = number of attributes
    output: cooc - n x n matrix, upper-right and lower-left triangles identical
    '''

    cooc = np.zeros((labels.shape[1], labels.shape[1]))
    for n in range(labels.shape[1]):
        for a in range(n,labels.shape[1]):
            # intersection of label confidences for the two attributes
            cooc[n, a] = np.sum([ min(labels[row,n],labels[row, a]) for row in range(labels.shape[0]) if labels[row,n] > 0.0])
            cooc[a, n] = cooc[n, a] 
        
    return cooc

def occur_mat(labels):
    '''
    create vector of occurence counts for set of features given training set
    input: labels - m x n matrix, m = number of instances, n = number of attributes
    output: occur - 1 x n matrix, counts for each dimension in trining set
    '''

    oc = np.sum(labels, axis=0)
    
        
    return oc

def get_next_query_ind(cur_label, train, occur_mat_train, ela_type, prev_label = None, force_alt_method = False, thresh = -1.0):
    '''
    follow occurcurence graph through all questions until one is unlabeled. 
    if finished output = -1
    input, unfished label vector < cur_label > type ndarray, training set < train > type ndarray

    if thresh is specified, next ind based on unlabed attr with highest MLE given set of labeled dimensions
          if no matching rows exist, 1) pick randomly from remaining labels 
          2) pick nth most popular dimension from rows matching prev_label not including dims that were labeled in cur_label
          3) find the self of closest rows by cosine dist and pick max MLE dimension from those rows
    
    '''

    assert len(cur_label) == train.shape[1], \
        "length of label vector %d and occurcurence matrix %d are not equal" % (len(cur_label), train.shape[1])
    
    new_match = 1

    # fill block list 
    block_list = np.where(cur_label != -1)[0]

    # pre_ind should only include conditional matching attributes
    if force_alt_method:
        mrows = np.array([])
    else:
        mrows = match_rows(cur_label, train, block_list)

    if mrows.shape[0] < 3:
        # print 'ran out of matching instances in training set or forcing alt method'
        if ela_type == 'rand':
            idx = random.choice([x for x in range(len(cur_label)) if x not in block_list])
            mle = np.true_divide(np.sum(train[:,idx]), train.shape[0])
            return idx, mle, new_match
        elif ela_type == 'pop':
            mrows = train
            occur = occur_mat_train
        elif ela_type == 'backoff':
            new_match = 0
            backoff_cur_label, orig_block_list, occur, mrows = get_occur_backoff(cur_label, prev_label, train)
        elif ela_type == 'dist':
            mrows = closest_rows(cur_label, train, block_list, 100)
            occur = occur_mat(mrows)
        else:
            print 'incorrent ela_type '+str(ela_type)
            return -1, 0.0. new_match
    else:
        occur = occur_mat(mrows)

    if new_match == 0:
        pre_ind = [i for i in range(len(cur_label)) if i not in block_list]
        idx, occur_cnt = get_idx_backoff(backoff_cur_label, orig_block_list, occur, pre_ind)

    else:
        pre_ind = [i for i in range(cur_label.shape[0]) if i not in block_list]
        idx = np.where(occur == occur[pre_ind].max())[0]
        idx = [ix for ix in idx if ix not in block_list]
        idx = idx[0]
        occur_cnt = occur[pre_ind].max()

    #Maximum Likelihood Estimate
    mle = np.true_divide(occur_cnt,mrows.shape[0])
    # print 'mle for idx %d = %.4f' % (idx,mle)

    # if this attribute is very unlikely, instead ask for the most likely attribute in alternative ela selection method
    # if this was already the alternative query, don't bother just return
    eps = 0.0001
    if mle < thresh+eps and not force_alt_method:
        # print 'mle %.4f < thresh+eps %.4f : retrying with alt method' % (mle, thresh+eps)
        update_cur_label = [x for x in cur_label]
        update_cur_label[idx] = 0
        force_alt_method = True
        idx, mle, new_match = get_next_query_ind(update_cur_label, train, occur_mat_train, ela_type, prev_label, force_alt_method, thresh)

    # TODO: heirarchical prior adjustment
        
    return idx, mle, new_match

def get_occur_backoff(cur_label, prev_label, train):
    if prev_label == None:
        print 'need a last known label for backoff method'
        return -1, 0, 0
    orig_block_list = np.where(cur_label != -1)[0]
    backoff_cur_label = [x for x in prev_label]
    block_list = np.where(prev_label != -1)[0]
    # print 'len block_list = '+str(len(block_list))
    # print [prev_label[x] for x in range(len(prev_label)) if x in block_list]
    mrows = match_rows(prev_label, train, block_list)
    occur = occur_mat(mrows)
    # print all(backoff_cur_label == prev_label)
    # print mrows.shape
    # print 'occur shape ' + str(occur.shape)
    return backoff_cur_label, orig_block_list, occur, mrows

def get_idx_backoff(cur_label, orig_block_list, occur, pre_ind):
    pre_ind = [x for x in range(len(cur_label)) if x not in orig_block_list]
    # print len(orig_block_list)
    # print pre_ind
    occur_vals = occur[pre_ind]
    # print occur_vals
    nth_largest = heapq.nlargest(1, range(len(pre_ind)), key = lambda x: occur_vals[x])        
    # print 'nth_largest : '+str(nth_largest)
    # print 'pre_ind[nth_largest] : '+str([pre_ind[n] for n in nth_largest])
    nth_largest = nth_largest[0]
    idx = pre_ind[nth_largest]
    occur_cnt = occur[idx]
    # print 'occur_cnt '+str(occur_cnt)
    return idx, occur_cnt

def check_dtree(dtree, knowns):
    if knowns in dtree.keys():
        return dtree[knowns]
    else:
        return -1

def update_dtree(dtree, new, knowns, mle):
    dtree[knowns] = (new, mle)

def get_knowns(cur_label):
    inds = np.where(cur_label != -1)[0]
    vals = cur_label[inds]
    knowns = str(zip(inds, vals))
    return knowns
