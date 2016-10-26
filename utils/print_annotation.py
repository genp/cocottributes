#!/usr/bin/env python
from PIL import Image, ImageDraw
from io import BytesIO
import json
import os
import requests

from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np

def print_image_with_attributes(img, attrs, category, sname):

    fig = plt.figure()
    plt.imshow(img)
    plt.axis('off')  # clear x- and y-axes
    plt.title(category)
    for ind, a in enumerate(attrs):
        plt.text(min(img.shape[1]+10, 1000), (ind+1)*img.shape[1]*0.1, a, ha='left')
    
    fig.savefig(sname, dpi = 300,  bbox_inches='tight')    


def print_coco_attributes_instance(cocottributes, coco_data, ex_ind, sname):
    # List of COCO Attributes
    attr_details = sorted(cocottributes['attributes'], key=lambda x:x['id'])
    attr_names = [item['name'] for item in attr_details]

    # COCO Attributes instance ID for this example
    coco_attr_id = cocottributes['ann_vecs'].keys()[ex_ind]

    # COCO Attribute annotation vector, attributes in order sorted by dataset ID
    instance_attrs = cocottributes['ann_vecs'][coco_attr_id]

    # Print the image and positive attributes for this instance, attribute considered postive if worker vote is > 0.5
    pos_attrs = [a for ind, a in enumerate(attr_names) if instance_attrs[ind] > 0.5]
    coco_dataset_ann_id = cocottributes['patch_id_to_ann_id'][coco_attr_id]

    coco_annotation = [ann for ann in coco_data['annotations'] if ann['id'] == coco_dataset_ann_id][0]

    img_url = 'http://mscoco.org/images/{}'.format(coco_annotation['image_id'])
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    polygon = coco_annotation['segmentation'][0]
    ImageDraw.Draw(img, 'RGBA').polygon(polygon, outline=(255,0,0), fill=(255,0,0,50))
    img = np.array(img)
    category = [c['name'] for c in coco_data['categories'] if c['id'] == coco_annotation['category_id']][0]

    print_image_with_attributes(img, pos_attrs, category, sname)


# Load COCO Dataset
data_types = ['val2014', 'train2014']
coco_data = {}
# Change this to location where COCO dataset lives
coco_dataset_dir = '/Users/gen/coco/'
for dt in data_types:
    annFile=os.path.join(coco_dataset_dir, 'instances_%s.json'%(dt))

    with open(annFile, 'r') as f:
        tmp = json.load(f)
        if coco_data == {}:
            coco_data = tmp
        else:
            coco_data['images'] += tmp['images']
            coco_data['annotations'] += tmp['annotations']

# Load COCO Attributes 
cocottributes = joblib.load('/Users/gen/Downloads/cocottributes_eccv_version.jbl')

# Index of example instance to print
ex_inds = [0,10,50,100,500,1000,5000,10000]

sname = 'example_cocottributes_annotation{}.jpg'
for ex_ind in ex_inds:
    print_coco_attributes_instance(cocottributes, coco_data, ex_ind, sname.format(ex_ind))
