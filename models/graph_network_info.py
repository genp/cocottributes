#!/usr/bin/env python

###
# Pull wieghts out of model snapshots and graph
# 
###

# import some modules
import sys, os, time
import os.path as osp

# TODO: change this to your own installation
caffe_root = '/home/gen/caffe/'  
sys.path.append(caffe_root+'python')

from sklearn.externals import joblib
import argparse
import caffe
import numpy as np

from sklearn.externals import joblib
import argparse

from models import print_funcs


# initialize caffe for gpu mode
caffe.set_mode_gpu()
caffe.set_device(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pull out weights from snapshots of network")
    parser.add_argument("-s", "--start", help="starting snapshot iteration number", type=int)
    parser.add_argument("-t", "--stop", help="stopping snapshot iteration number", type=int)
    parser.add_argument("-i", "--iter", help="iteration step size", type=int)
    parser.add_argument("--snapshot_dir", help="caffe model params file", type=str)
    parser.add_argument("--save_dir", help="location to save weight files", type=str)
    args = parser.parse_args()
    os.chdir(args.snapshot_dir)
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # Objects for logging solver training
    _train_loss = []
    _weight_params = {}


    start = args.start
    stop = args.stop
    timestr = 'snaps_{0}_{1}'.format(start, stop)
    for itt in range(start, stop+1, args.iter):

        # TODO: change this to the name of your default solver file and shapshot file
        solver = caffe.SGDSolver(osp.join(args.snapshot_dir, 'solver.prototxt'))
        solver.restore(osp.join(args.snapshot_dir, 'snapshot_iter_{}.solverstate'.format(itt)))

        solver.net.forward()
        _train_loss.append(solver.net.blobs['loss'].data) # this should be output from loss layer
        print_funcs.print_layer_params(solver, _weight_params)
        print '******************************************************** Loss train for iter {0}: {1}'.format(itt, _train_loss)
        joblib.dump(_weight_params, osp.join(args.save_dir, 'network_parameters_%s.jbl' % timestr), compress=6)
        joblib.dump(_train_loss, osp.join(args.save_dir, 'network_loss_%s.jbl'% timestr), compress=6)
        
        del solver
