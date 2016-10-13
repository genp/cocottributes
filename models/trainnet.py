import cocottributes_mutltilabel_net
import cocottributes_tools as tools #this contains some tools that we need

# write train and val nets.
with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
    # provide parameters to the data layer as a python dictionary. Easy as pie!
    # TODO change imsize to 128 - correction for 11x11 conv1 filters...
    data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'train2014', label_ids = label_ids)
    f.write(caffenet_multilabel(data_layer_params, 'CocottributesMultilabelDataLayerSync', num_labels))

with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
    data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'val2014', label_ids = label_ids)
    f.write(caffenet_multilabel(data_layer_params, 'CocottributesMultilabelDataLayerSync', num_labels))

# Objects for logging solver training
_train_loss = []
_weight_params = {}    

solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
#solver.net.copy_from(osp.join(workdir, 'snapshot_iter_101.caffemodel'))
solver.test_nets[0].share_with(solver.net)
solver.snapshot()

t0 = time.clock()
solver.step(1)
print time.clock() - t0, "seconds process time"


_train_loss.append(solver.net.blobs['loss'].data) # this should be output from loss layer
print_funcs.print_layer_params(solver, _weight_params)
timestr = time.strftime("%Y%m%d-%H%M%S")
joblib.dump(_weight_params, osp.join(workdir, 'plots/cocottributes_network_parameters_%s.jbl' % timestr), compress=6)

from cocottributes_tools import SimpleTransformer
from copy import copy
transformer = SimpleTransformer() # this is simply to add back the bias, re-shuffle the color channels to RGB, and so on...

image_index = 0 #Lets look at the first image in the batch.
#plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
sname = '/gpfs/main/home/gen/coco_attributes/scratch/train_img.jpg'
plt.imsave(sname , transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)

#Load classes for printing gt and estimated labels
classes = [x.name for x in Label.query.filter(Label.parent_id == obj_attr_supercategory_id).order_by(Label.id).all()]
print 'Num attributes: %d' % len(label_ids)
print 'Ground truth: ',
for idx, val in enumerate(gtlist):
    if val:
        print classes[idx] + ',',
print ''



def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)
# This is checking the baseline if this classifier says Negative to everything        
def check_baseline_accuracy(net, num_batches, num_labels, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = np.zeros((batch_size, num_labels))
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)




from sklearn.metrics import average_precision_score, accuracy_score

def check_ap(net, num_batches, batch_size = 128):
    ap = np.zeros((len(label_ids),1))
    baseline_ap = np.zeros((len(label_ids),1))
    for n in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data
        baseline_ests = -1*np.ones(gts.shape) 
        for dim in range(gts.shape[1]):
            tmp = gts[:,dim]
            fmt_gt = tmp[np.where(tmp!=-1)]
            fmt_gt[np.where(fmt_gt==0)] = -1
            fmt_est = ests[:,dim]
            fmt_est = fmt_est[np.where(tmp!=-1)]
            fmt_est_base = baseline_ests[:,dim]
            fmt_est_base = fmt_est_base[np.where(tmp!=-1)]            
            ap_score = average_precision_score(fmt_gt, fmt_est)
            base_ap_score = average_precision_score(fmt_gt, fmt_est_base)
            #print classes[dim] +' ' +str(ap_score)
            ap[dim] = ap_score
            baseline_ap[dim] = base_ap_score

    return ap/float(num_batches), baseline_ap/float(num_batches)


ap, baseline_ap = check_ap(solver.test_nets[0], 5)
ap_scores = {}
ap_scores['ap'] = ap
ap_scores['baseline_ap'] = baseline_ap
print '*** Mean AP and Baseline AP scores {}***'.format(0)
print np.mean(ap)
print np.mean(baseline_ap)
joblib.dump(ap_scores, osp.join(workdir, 'plots/ap_scores_%d.jbl' % -1))
solver.snapshot()

for itt in range(100):#500):
    solver.step(1)
    _train_loss.append(solver.net.blobs['loss'].data) # this should be output from loss layer
    print_funcs.print_layer_params(solver, _weight_params)

    if itt % 10 == 0: # 100 not 1
        ap, baseline_ap = check_ap(solver.test_nets[0], 5)
        ap_scores = {}
        ap_scores['ap'] = ap
        ap_scores['baseline_ap'] = baseline_ap
        print '*** Mean AP and Baseline AP scores {}***'.format(itt)
        print np.mean([b if not np.isnan(b) else 0 for b in ap])
        print np.mean([b if not np.isnan(b) else 0 for b in baseline_ap])
        joblib.dump(ap_scores, osp.join(workdir, 'plots/ap_scores_%d.jbl' % itt))
        solver.snapshot()
        
    joblib.dump(_weight_params, osp.join(workdir, 'plots/cocottributes_network_parameters_%s.jbl' % timestr), compress=6)
    joblib.dump(_train_loss, osp.join(workdir, 'plots/cocottributes_network_loss_%s.jbl'% timestr), compress=6)

baseline_ap = check_baseline_ap(solver.test_nets[0], len(val_ids)/128, num_labels)
ap = check_ap(solver.test_nets[0], len(val_ids)/128)
ap_scores = {}
ap_scores['ap'] = ap
ap_scores['baseline_ap'] = baseline_ap
print '*** Mean AP and Baseline AP scores {}***'.format(itt)
print np.mean(ap)
print np.mean(baseline_ap)
joblib.dump(ap_scores, osp.join(workdir, 'plots/ap_scores_%d.jbl' % itt))
    
print 'itt:{}'.format(itt), 'accuracy:{0:.4f}'.format(check_accuracy(solver.test_nets[0], 2))
print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], len(val_ids)/128, num_labels))

image_index = 0 #Lets look at the first image in the batch.
# test_net = solver_async.test_nets[0]
test_net = solver.test_nets[0]
test_net.forward()
#plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
sname = '/gpfs/main/home/gen/coco_attributes/scratch/test_img.jpg'
plt.imsave(sname , transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
estlist = test_net.blobs['score'].data[image_index, ...] > 0
print 'Ground truth: ',
for idx, val in enumerate(gtlist):
    if val == 1:
        print classes[idx] + ',',

print ''
print 'Estimated: ',
for idx, val in enumerate(estlist):
    if val == 1:
        print classes[idx] + ',',                                                                
