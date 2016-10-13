#!/usr/bin/env python
import os
import sys
sys.path.append('/home/gen/caffe/python')
import argparse

from sklearn.externals import joblib
import caffe
import numpy as np

sys.path.append('../')
from app import db
from app.models import Patch, Image, Feature, Label
# model setup
MODEL_FILE = '/home/gen/caffe/models/hybridCNN/hybridCNN_deploy_FC7.prototxt'
# pretrained weights
PRETRAINED = '/home/gen/caffe/models/hybridCNN/hybridCNN_iter_700000.caffemodel'

def norm_convnet_feat(feature):
    D = 200
    try:
        feature = feature[0][:D]
    except IndexError, e:
        feature = feature[:D]
    
    # power norm
    alpha = 2.5

    feature = [np.power(f, alpha) if f > 0.0 else -1.0*np.power(np.abs(f), alpha) for f in feature]
    norm = np.linalg.norm(feature)
    feature = np.array([f/norm for f in feature])

    return feature


def img_feat(img_ids, net, transformer, save_path, type_name, is_patch=False, normalize=True):

    img_arrays = []
    savenames = {}
    savenames['fc7'] = []
    # savenames['fc6'] = []
    # good_img_ids = [] 
    feat = {}
    feat['fc7'] = []
    # feat['fc6'] = []
    normsavenames = []
    for cnt, idx in enumerate(img_ids):
        print cnt
        check = Feature.query.filter(Feature.patch_id == idx).filter(Feature.type == type_name+'_fc7').first()
        if check != None:
            print 'skipping' 
            continue
        subdir = str(idx)[:2]
        if not os.path.exists(os.path.join(save_path, subdir)):
            os.makedirs(os.path.join(save_path, subdir))
            os.makedirs(os.path.join(save_path, subdir, 'norm'))            
        sname = os.path.join(save_path, subdir, 'img_%d' % idx)
        if not os.path.exists(sname):
            savenames['fc7'].append(sname+'_fc7.jbl')
            # savenames['fc6'].append(sname+'_fc6.jbl')            
            # good_img_ids.append(idx)
        else:
            continue


        if is_patch:
            img = Patch.query.get(idx).crop(savename = None, make_square = True)
        else:
            img = Image.query.get(idx).to_array()
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1).repeat(3,2)        

        net.blobs['data'].data[...] = transformer.preprocess('data',img)
        out = net.forward(blobs=['fc6'])
        feat['fc7'].append(out['fc7'])
        # feat['fc6'].append(out['fc6'])

        try:
            joblib.dump(feat['fc7'][-1], savenames['fc7'][-1])
            if is_patch:
                f = Feature(type = type_name+'_fc7', location = savenames['fc7'][-1], patch_id = idx)
            else:
                f = Feature(type = type_name+'_fc7', location = savenames['fc7'][-1], image_id = idx)
            db.session.add(f)
            # joblib.dump(feat['fc6'][-1], savenames['fc6'][-1])
            # if is_patch:
            #     f6 = Feature(type = type_name+'_fc6', location = savenames['fc6'][-1], patch_id = idx)
            # else:
            #     f6 = Feature(type = type_name+'_fc6', location = savenames['fc6'][-1], image_id = idx)
            # db.session.add(f6)
            db.session.commit()
        
            print 'idx %d' % idx
            sname = os.path.join(save_path, subdir, 'norm', 'img_%d' % idx)
            if not os.path.exists(sname):
                normsavenames.append(sname)
            else:
                continue

            print normsavenames[-1]+'_fc7.jbl'
            joblib.dump(norm_convnet_feat(feat['fc7'][-1]), normsavenames[-1]+'_fc7.jbl')
            if is_patch:
                f2 = Feature(type = type_name+'_fc7_norm', location = normsavenames[-1]+'_fc7.jbl', patch_id = idx)
            else:
                f2 = Feature(type = type_name+'_fc7_norm', location = normsavenames[-1]+'_fc7.jbl', image_id = idx)    
            db.session.add(f2)
            # joblib.dump(norm_convnet_feat(feat['fc6'][-1]), normsavenames[-1]+'_fc6.jbl')
            # if is_patch:
            #     f3 = Feature(type = type_name+'_fc6_norm', location = normsavenames[-1]+'_fc6.jbl', patch_id = idx)
            # else:
            #     f3 = Feature(type = type_name+'_fc6_norm', location = normsavenames[-1]+'_fc6.jbl', image_id = idx)    
            # db.session.add(f3)
            db.session.commit()
        except:
            print "Unexpected error:", sys.exc_info()
            print 'something broke with this pid/img_id', idx

        
    # for jdx, img_id in enumerate(good_img_ids):
    #     joblib.dump(feat['fc7'][jdx], savenames['fc7'][jdx])
    #     f = Feature(type = type_name+'_fc7', location = savenames['fc7'][jdx], image_id = img_id)
    #     db.session.add(f)
    #     joblib.dump(feat['fc6'][jdx], savenames['fc6'][jdx])
    #     f6 = Feature(type = type_name+'_fc6', location = savenames['fc6'][jdx], image_id = img_id)
    #     db.session.add(f6)
    #     db.session.commit()
        
    # if normalize:
    #     normsavenames = []
    #     for idx in img_ids:
    #         print 'idx %d' % idx
    #         sname = os.path.join(save_path,'norm', 'img_%d' % idx)
    #         if not os.path.exists(sname):
    #             normsavenames.append(sname)
    #         else:
    #             continue

    #     for jdx, image_id in enumerate(good_img_ids):
    #         joblib.dump(norm_convnet_feat(feat['fc7'][jdx]), normsavenames[jdx]+'_fc7.jbl')
    #         if is_patch:
    #             f2 = Feature(type = type_name+'_fc7_norm', location = normsavenames[jdx]+'_fc7.jbl', patch_id = image_id)
    #         else:
    #             f2 = Feature(type = type_name+'_fc7_norm', location = normsavenames[jdx]+'_fc7.jbl', image_id = image_id)    
    #         db.session.add(f2)
    #         joblib.dump(norm_convnet_feat(feat['fc6'][jdx]), normsavenames[jdx]+'_fc6.jbl')
    #         if is_patch:
    #             f3 = Feature(type = type_name+'_fc6_norm', location = normsavenames[jdx]+'_fc6.jbl', patch_id = image_id)
    #         else:
    #             f3 = Feature(type = type_name+'_fc6_norm', location = normsavenames[jdx]+'_fc6.jbl', image_id = image_id)    
    #         db.session.add(f3)
    #         db.session.commit()
        
    
    
# def patch_feat(patch_ids, net, transformer, save_path, type_name, normalize=True):

#     img_arrays = []
#     savenames = []
#     good_patch_ids = []
#     for idx in patch_ids:
#         print 'idx %d' % idx
#         sname = os.path.join(save_path, 'patch_%d.jbl' % idx)
#         if not os.path.exists(sname):
#             savenames.append(sname)
#             good_patch_ids.append(idx)
#         else:
#             continue
#         bbox = Patch.query.get(idx).crop(savename = None, make_square = True, resize = (227,227))
#         if len(bbox.shape) == 2:
#             bbox = bbox.reshape(bbox.shape[0], bbox.shape[1], 1).repeat(3,2)
#         print 'bbox shape: '+str(bbox.shape)
#         img_arrays.append(bbox)

#     print len(img_arrays)
#     feat = classifier.predict(img_arrays)

#     for jdx, patch_id in enumerate(good_patch_ids):
#         joblib.dump(feat[jdx], savenames[jdx])
#         f = Feature(type = type_name, location = savenames[jdx], patch_id = patch_id)
#         db.session.add(f)

#     if normalize:
#         normsavenames = []
#         for idx in patch_ids:
#             print 'idx %d' % idx
#             sname = os.path.join(save_path,'norm', 'patch_%d.jbl' % idx)
#             if not os.path.exists(sname):
#                 normsavenames.append(sname)
#             else:
#                 continue

#         for jdx, patch_id in enumerate(good_patch_ids):
#             joblib.dump(norm_convnet_feat(feat[jdx]), normsavenames[jdx])
#             f2 = Feature(type = type_name+'_norm', location = normsavenames[jdx], patch_id = patch_id)    
#             db.session.add(f2)
        
#     db.session.commit()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make caffe features for given image or patch id")
    parser.add_argument("-i", "--image_id", help="", type=int, nargs='*')
    parser.add_argument("-p", "--patch_id", help="", type=int, nargs='*')
    parser.add_argument("--feat_type", help="name of feature", type=str)    
    parser.add_argument("--model_file", help="caffe model params file", type=str)
    parser.add_argument("--pretrained", help="caffe pretrained model file", type=str)
    parser.add_argument("--save_dir", help="location to save features", type=str)

    args = parser.parse_args()

    if args.model_file:
        MODEL_FILE = args.model_file
    if args.pretrained:
        PRETRAINED = args.pretrained
        
    # classifier = caffe.Classifier(MODEL_FILE, PRETRAINED)
    net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load('/home/gen/caffe/models/hybridCNN/hybridCNN_mean.npy').mean(1).mean(1))
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))
    net.blobs['data'].reshape(1,3,227,227)

    ###### TODO: temp lines
    caffe.set_mode_gpu()
    # from app import db
    # attrs = [x.id for x in Label.query.filter(Label.parent_id == 407).all()] + \
    #         [x.id for x in Label.query.filter(Label.parent_id == 102).all()]
    # from mturk import manage_hits
    # patch_ids = []
    # img_ids = []
    # cat_ids = [[1,3], [1,18], [1,54], [1,56], [1,17], [1,20], [25], [1,4], [1,2]]
    # for cat in cat_ids:
    #     # pids = manage_hits.find_patches(cat, attrs, [])
    #     ims = manage_hits.find_images(cat, attrs, [])[:50]
    #     print 'img id', cat, ims[0]
    #     img_ids += ims
    #     patch_ids += [x.id for i in ims for x in Image.query.get(i).patches]
    #     print 'num patches', len(patch_ids)
    # ids = {}
    # ids['patch_ids'] = patch_ids
    # ids['img_ids'] = img_ids
    # joblib.dump(ids, '/home/gen/coco_attributes/data/test_ids.jbl')
    # test_ids = joblib.load('/home/gen/coco_attributes/data/test_ids.jbl')
    # img_ids = test_ids['img_ids']
    # patch_ids = test_ids['patch_ids']
    # print 'calculating features for imgs', len(img_ids), args.feat_type
    stmt = "select patch_id from (select a.patch_id, count(*) from annotation a, label lbl where a.label_id = lbl.id and lbl.parent_id = 407 group by a.patch_id) as tmp where count > 60"
    patch_ids = [x[0] for x in db.engine.execute(stmt).fetchall()]
    print 'calculating features for patch', len(patch_ids), args.feat_type    
    # img_feat(img_ids, net, transformer, args.save_dir, args.feat_type)
    for group in range(len(patch_ids)/1000+1):
        group_ids = patch_ids[group*1000:min((group+1)*1000,len(patch_ids))]
        img_feat(group_ids, net, transformer, args.save_dir, args.feat_type, True)
    ######
        
    if args.image_id:
        img_feat(args.image_ids, net, transformer, args.save_dir, args.feat_type) 
        
    if args.patch_id:
        img_feat(args.patch_id, net, transformer, args.save_dir, args.feat_type, True)
