import json, time, pickle, scipy.misc, skimage.io, caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from cocottributes_tools import SimpleTransformer

# from app.models import Patch
from sklearn.externals import joblib

class CocottributesMultilabelDataLayerSync(caffe.Layer):
    """
    This is a simple syncronous datalayer for training a multilabel model on Cocottributes. 
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str) 

        # do some simple checks that we have the parameters we need.
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'split' in params.keys(), 'Params must include Cocottributes database patch instance ids. split (train, val, or test).'
        assert 'im_shape' in params.keys(), 'Params must include im_shape.'
        assert 'num_labels' in params.keys(), 'Params must include number of labels.'
        assert 'dataset_file' in params.keys(), 'Params must include Cocottributes dataset file.'
        
        # store input as class variables
        self.batch_size = params['batch_size'] 
        self.im_shape = params['im_shape']
        # uses cocottributes dataset indices
        self.num_labels = params['num_labels']
        self.indexlist = params['split']
        self.labels = joblib.load(params['dataset_file'])
        
        self._cur = 0 # current image
        self.transformer = SimpleTransformer() #this class does some simple data-manipulations

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1]) # since we use a fixed input image size, we can shape the data layer once. Else, we'd have to do it in the reshape call.
        top[1].reshape(self.batch_size, len(self.num_labels))

        print "CocottributesMultilabelDataLayerSync initialized for split: {}, with bs:{}, im_shape:{}, and {} images.".format(params['split'], params['batch_size'], params['im_shape'], len(self.indexlist))


    def reshape(self, bottom, top):
        """ no need to reshape each time sine the input is fixed size (rows and columns) """
        pass 

    def forward(self, bottom, top):
        """
        Load data. 
        """
        for itt in range(self.batch_size):

            # Did we finish an epoch?
            if self._cur == len(self.indexlist):
                self._cur = 0
                shuffle(self.indexlist)
            
            # Load an image and prepare ground truth
            index = self.indexlist[self._cur] # Get the image index
            im, multilabel = load_cocottributes_annotation(index, self.im_shape, self.labels)

            # do a simple horizontal flip as data augmentation
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]
            
            # Add directly to the caffe data layer
            top[0].data[itt, ...] = self.transformer.preprocess(im)
            top[1].data[itt, ...] = multilabel
            self._cur += 1

    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass




class CocottributesMultilabelDataLayerAsync(caffe.Layer):
    """
    This is a simple asyncronous datalayer for training a multilabel model on COCO. 
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str) 

        # do some simple checks that we have the parameters we need.
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        assert 'im_shape' in params.keys(), 'Params must include im_shape.'
        assert 'num_labels' in params.keys(), 'Params must include number of labels'
        assert 'dataset_file' in params.keys(), 'Params must include Cocottributes dataset file.'
                
        self.batch_size = params['batch_size'] # we need to store this as a local variable.
        self.im_shape = params['im_shape']
        # uses cocottributes dataset indices
        self.num_labels = params['num_labels']
        self.indexlist = params['split']        
        self.labels = joblib.load(params['dataset_file'])                
        # === We are going to do the actual data processing in a seperate, helperclass, called BatchAdvancer. So let's forward the parame to that class ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = BatchAdvancer(self.thread_result, params)
        self.dispatch_worker() # Let it start fetching data right away.

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, params['im_shape'][0], params['im_shape'][1]) # since we use a fixed input image size, we can shape the data layer once. Else, we'd have to do it in the reshape call.
        top[1].reshape(self.batch_size, len(self.num_labels)) 

        print "CocottributesMultilabelDataLayerAsync initialized for split: {}, with bs:{}, im_shape:{}.".format(params['split'], params['batch_size'], params['im_shape'])



    def reshape(self, bottom, top):
        """ no need to reshape each time sine the input is fixed size (rows and columns) """
        pass 

    def forward(self, bottom, top):
        """ this is the forward pass, where we load the data into the blobs. Since we run the BatchAdvance asynchronously, we just wait for it, and then copy """

        if self.thread is not None:
            self.join_worker() # wait until it is done.

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] #Copy the already-prepared data to caffe.
        
        self.dispatch_worker() # let's go again while the GPU process this batch.

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class BatchAdvancer():
    """
    This is the class that is run asynchronously and actually does the work.
    """
    def __init__(self, result, params):
        self.result = result
        self.batch_size = params['batch_size'] 
        self.im_shape = params['im_shape']
        # TODO:
        self.num_labels = params['num_labels']
        self.indexlist = params['split'] 
        self._cur = 0 # current image
        self.transformer = SimpleTransformer() #this class does some simple data-manipulations

        print "BatchAdvancer initialized with {} images".format(len(self.indexlist))

    def __call__(self):
        """
        This does the same stuff as the forward layer of the synchronous layer. Exept that we store the data and labels in the result dictionary (as lists of length batchsize).
        """
        self.result['data'] = []
        self.result['label'] = []
        for itt in range(self.batch_size):

            # Did we finish an epoch?
            if self._cur == len(self.indexlist):
                self._cur = 0
                shuffle(self.indexlist)

            # Load an image and prepare ground truth
            index = self.indexlist[self._cur] # Get the image index
            im, multilabel = load_cocottributes_annotation(index, self.im_shape, self.labels)
                        
            # do a simple horizontal flip as data augmentation
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]

            # Store in a result list.
            self.result['data'].append(self.transformer.preprocess(im))
            self.result['label'].append(multilabel)
            self._cur += 1


# def load_cocottributes_annotation(idx, im_size, label_ids):
#     """
#     Load MS COCO Attribute label given Cocottributes database instance (patch) id
#     """
#     patch= Patch.query.get(idx)
#     im = patch.crop(savename = None, make_square = True, resize = im_size, local = True)
#     ann_vec,_ = patch.annotation_vector(label_ids, consensus=True)
#     multilabel = np.array([1 if x >= 0.5 else -1 if x == -1 or (x > 0 and x < 0.5) else 0 for x in ann_vec])
#     return im, multilabel

def load_cocottributes_annotation(idx, im_size, labels):
    """
    Load MS COCO Attribute label given Cocottributes database instance (patch) id
    """
    patch= Patch.query.get(idx)
    im = patch.crop(savename = None, make_square = True, resize = im_size, local = True)
    ann_vec = labels['ann_vecs'][idx] 
    multilabel = np.array([1 if x >= 0.5 else -1 if x == -1 or (x > 0 and x < 0.5) else 0 for x in ann_vec])
    return im, multilabel
