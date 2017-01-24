# Reference http://deepdish.io/2015/04/28/creating-lmdb-in-python/
# https://groups.google.com/forum/#!topic/caffe-users/RuT1TgwiRCo

import caffe
from caffe.proto import caffe_pb2
import lmdb
import sys
import os
import numpy as np
import scipy as scp
from scipy import misc, ndimage
import utils
from tqdm import tqdm


def create_image_lmdb(image_data, split="train", shape=(3, 224, 224), data_root=None, crop_size=224):
    """
    Create the LMDB for the images.
    :param image_data: List of JSON objects with details about the images.
    :param split: Specify which data split this LMDB is for.
    :param shape: The shape of the image with which to store in the LMDB.
    :param data_root: The root folder where to store the LMDB file.
    :param crop_size: The final size of the cropped image.
    :return:
    """
    print("Creating {0} Image LMDB".format(split))
    channels, width, height = shape
    filename = os.path.join(data_root, "image-{0}-lmdb".format(split))

    # We need to prepare the database for the size. We'll set it 100 times
    # greater than what we theoretically need.
    map_size = len(image_data) * channels * width * height * 100

    in_db = lmdb.open(filename, map_size=map_size)
    with in_db.begin(write=True) as in_txn:
        for idx, ann in tqdm(enumerate(image_data), total=len(image_data)):
            # load image:
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)

            img = scp.ndimage.imread(ann["path"])

            # Crop out the object with context padding.
            x, y, width, height = ann["bbox"]
            crop_im = utils.get_image_crop(img, x, y, width, height, crop_size)

            # Resize it to the desired shape.
            im = scp.misc.imresize(crop_im, (crop_size, crop_size, 3))  # resize

            im = im[:, :, ::-1]
            im = im.transpose((2, 0, 1))

            im_data = caffe.io.array_to_datum(im)

            key = '{:0>7d}'.format(idx)
            # if using python3, we need to convert the UTF-8 string to a byte array
            if sys.version_info[0] == 3:
                key = key.encode('utf-8')

            in_txn.put(key, im_data.SerializeToString())

            # print("Saving {0}".format(image_data))

    in_db.close()


def create_label_lmdb(image_data, split="train", data_root=None):
    """
    Create the LMDB for the labels of the dataset.
    :param image_data: The list of JSON for each image datum containing a key called
    :param split:
    :param data_root:
    :return:
    """
    print("Creating {0} label LMDB".format(split))
    filename = os.path.join(data_root, "label-{0}-lmdb".format(split))
    map_size = len(image_data) * 204 * 100

    in_db = lmdb.open(filename, map_size=map_size)
    with in_db.begin(write=True) as in_txn:
        for idx, ann in tqdm(enumerate(image_data), total=len(image_data)):

            # Get the object attributes as an array of 1s and 0s.
            img_attrs = ann['attrs_vector']
            # print(img_attrs)

            # Since caffe array_to_datum requires a 3-dim array, we reshape the attribute vector.
            img_attrs = img_attrs.reshape(list(img_attrs.shape) + [1, 1])

            data = caffe.io.array_to_datum(img_attrs)
            # print(data.SerializeToString())
            # print("Saving {0}".format(idx))

            key = '{:0>7d}'.format(idx)
            # if using python3, we need to convert the UTF-8 string to a byte array
            if sys.version_info[0] == 3:
                key = key.encode('utf-8')

            in_txn.put(key, data.SerializeToString())

    in_db.close()


def view_lmdb_image(lmdb_file, index):
    """
    Utility function to view image at position `index` in the lmdb_file
    after reversing various operations done during insertion.
    :param lmdb_file: The name of the LMDB directory which you wish to examine.
    :param index: The position of the image you wish to examine.
    :return: None, it just shows the image to you.
    """
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    for key, value in lmdb_cursor:
        if index > 0:
            index -= 1
            continue

        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)

        # CxHxW to HxWxC
        image = np.transpose(data, (1, 2, 0))
        # Reverse from BGR to RGB
        image = image[:, :, ::-1]
        scp.misc.imshow(image)


if __name__ == "__main__":
    # Example of how to use the LMDB helper functions.
    source_path = "../../../MSCOCO/annotations"

    from sklearn.externals import joblib
    import json
    import os.path as osp

    print("Reading the attributes")
    attr_data = joblib.load('../../../MSCOCO/cocottributes_new_version.jbl')

    print("Reading the annotation data")
    train_ann = json.load(open(osp.join(source_path, 'instances_{0}2014.json'.format('train'))))
    val_ann = json.load(open(osp.join(source_path, 'instances_{0}2014.json'.format('val'))))

    print("Getting the formatted image list")
    root = "../../../MSCOCO"
    im_data_list = utils.get_images_list(val_ann['annotations'], attr_data, data_root=root, split="val")
    # create_label_lmdb(im_data_list, split='sample', data_root=root)
    create_image_lmdb(im_data_list, split="sample", data_root=root)
