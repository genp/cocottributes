import copy
import time

import numpy as np



class SGraph:
    
    def __init__(self, labels, numq , dtree):
        self.dtree = dtree
        self.train = labels
        self.numq = numq
        self.labels = np.unique(self.train)
    
    def get_next_query(self, cur_label):
        # check dtree
        knowns = get_knowns(cur_label)
        check = check_dtree(self.dtree, knowns)

        if not check == -1:
            next_ind = check

        else:
            next_ind = get_next_query_ind(cur_label, self.train, self.labels)
            update_dtree(self.dtree,next_ind,knowns)
        next_ind = get_next_query_ind(cur_label, self.train, self.labels)
        return next_ind


    def test(self, true_label):
        '''
        follow version space reduction until single hypothesis determined
        '''
        est = -1*np.ones((1,len(true_label)))
        lidx = []
        for j in range(self.numq):
            idx = self.get_next_query(est)
            # print 'next ind :'
            # print idx
            lidx.append(idx)
            est[0,idx] = true_label[idx]
            # print 'true label, new label (%d,%d)' %(est[0,idx], true_label[idx])
            # print 'total pos labels: %d' % sum([tmp for tmp in est[0,:] if tmp != -1])

        # avg vectors with matching labeled values, if none exist back off
        match_inds = lidx
        while np.where(est == -1)[0].shape[0] > 0:
            # find all rows in train that match est on match_inds
            rows = match_rows(est, self.train, match_inds)
            # print rows
            # if these rows exist, avg and update est, break
            if not rows.shape[0] == 0:
                avg = [1 if elem > 0.5 else 0 for elem in np.mean(rows, axis=0)]
                est = [avg[j] if j not in lidx else est[0,j] for j in range(est.shape[1])]
            match_inds = match_inds[:-1]
            if not match_inds:
                print 'thing empty, big break'
                return -1
        num_tp = sum([elem for ind,elem in enumerate(est) if true_label[ind] == 1])
        pos = sum(est)
        if pos == 0:
            prec = 1.0
            print 'didnt estimate any true values for :'
            print true_label
        else:
            prec = num_tp/pos
        true_pos = sum(true_label)
        if true_pos == 0:
            rec = 1.0
        else:
            rec = num_tp/true_pos
        print ' prec, rec for this query = %f, %f ' % (prec, rec)
        return est, lidx, prec, rec


def match_rows(query, train, match_inds):
    # TODO: all check is bad
    # match_rows = np.array([row for i,row in enumerate(train) if all(row[match_inds] == query[match_inds])])    
    mesh = np.ix_(range(train.shape[0]), match_inds)
    subarray = query[:,match_inds]
    comp = (train[mesh] == subarray).all(axis=1)
    match_rows = train[comp]
    return match_rows


def get_next_query_ind(cur_label, train, labels):
    '''
    calculate dimension index that minimizes expected entropy of cur_label given training set
    if cur_label is finished output = -1
    input, unfished label vector < cur_label > type ndarray, training set < train > 
    '''
    assert cur_label.shape[1] == train.shape[1], \
        "length of label vector %d and cooccurence matrix %d are not equal" % (cur_label.shape[1], train.shape[1])

    possible_labels = [0,1]
    unlabeled_dim = np.where(cur_label== -1)[1]
    # print 'num unlabeled_dim in get next query= %d' % len(unlabeled_dim)
    exp_ent = []
    for x in unlabeled_dim:
        marginal_ent = calc_exp_ent(x, cur_label, train, labels)
        # print 'marginal_ent = %f' % marginal_ent
        exp_ent.append(marginal_ent)
        # print len(exp_ent)
    # print 'ind min exp_ent'
    # print exp_ent.index(min(exp_ent))
    idx = unlabeled_dim[exp_ent.index(min(exp_ent))]

    return idx

def calc_exp_ent(x, cur_label, train, labels):
    '''
    calculate expected entropy of unlabeled dimension x
    '''
    # print 'a = %d' %x
    tic = time.time()
    # print tic
    # print 'expected entropy of dim %d' % x
    unlabeled_dim = np.where(cur_label == -1)[1]
    exp_ent = 0.0
    unique_labels, counts_labels = np.unique(train[:,x], return_counts=True)
    # print unique_labels, counts_labels
    tot_count = float(np.sum(counts_labels))
    for cat in labels:
        # print time.time() - tic
        try:
            cat_ind = np.where(unique_labels == cat)[0][0]
            count = counts_labels[cat_ind]
        except IndexError, e:
            count = 0
        cond_prob = count/tot_count
        joint_ent = 0
        for dim in unlabeled_dim: # TODO:should this be all dim or all unlabeled dim??
            # print 'dim = %d' % dim
            # print time.time() - tic
            if dim == x:
                continue
            cond_ent = 0
            query = copy.deepcopy(cur_label)
            query[0,x] = cat
            # print 'deepcopy'
            # print time.time() - tic
            mr = match_rows(query, train, [x])
            # print 'sub mat' 
            # print time.time() - tic
            # TODO:this might not be the right thing. maybe want whole training set union with <query>
            ud, cd = np.unique(mr[:,dim], return_counts=True) 
            tot_cd = float(mr.shape[0])
            for val in labels:
                try:
                    val_ind = np.where(ud == val)[0][0]
                    c = cd[val_ind]
                except IndexError, e:
                    c = 0
                prob_label_given_future = c/tot_cd
                if prob_label_given_future == 0.0:
                    cond_ent = 0.0
                else:
                    cond_ent = -1*prob_label_given_future*np.log2(prob_label_given_future)
                # print 'k = %d, j = %d, cond_ent = %f' % (dim, val, cond_ent)
                joint_ent += cond_ent
        # print 'cat = %d, joint_ent = %f, cond_prob = %f' % (cat, joint_ent, cond_prob)
        exp_ent += cond_prob*joint_ent

    return exp_ent

def check_dtree(dtree, knowns):
    if knowns in dtree.keys():
        return dtree[knowns]
    else:
        return -1

def update_dtree(dtree, new, knowns):
    # print knowns
    # print type(knowns)
    dtree[knowns] = new

def get_knowns(cur_label):
    inds = np.where(cur_label != -1)[1]
    vals = cur_label[0,inds]
    knowns = str(zip(inds, vals))
    return knowns
