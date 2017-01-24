"""This module contains all the utilities needed to create the various HDF5 files."""
import h5py
import numpy as np
import scipy as scp
from scipy import ndimage, misc
import traceback
import os
import utils
from tqdm import tqdm


def create_dataset(image_data, filename="train.hdf5",
                   shape=(3, 224, 224), data_root="", crop_size=224):
    """
    Create a HDF5 file to store the data set, so we can use it with Keras or Blocks.
    Each image is the segment of the object for which the attribute applies.
    We store the images with the ordering in the first element of the shape tuple i.e. (idx, 3, x, y)
    Ref. https://www.getdatajoy.com/learn/Read_and_Write_HDF5_from_Python

    :param image_data: List of JSON objects with details about the images.
    :param filename: The filename for the HDF5 file. You should specify the split here.
    :param shape: The shape of the image to save into the HDF5 file.
    :param data_root: The source path of the MSCOCO dataset.
    :param crop_size: The final size of the cropped image.
    :return: We store the images with the ordering in the first element of the shape tuple i.e. (idx, 3, x, y)
    """
    # Ensure the extension of the filename is `hdf5`
    if filename.split('.')[-1].strip() != 'hdf5':
        filename += ".hdf5"

    filename = os.path.join(data_root, filename)

    try:
        # Get the shape of the HDF5 file based on the number of images in the split
        n_images = len(image_data)
        dataset_shape = (n_images,) + shape

        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset("data", dataset_shape)
            labels = f.create_dataset("attributes", (n_images, 204))
            metadata = f.create_dataset("metadata", (n_images, 1))

            dset.attrs["desc"] = "This is the image data."
            labels.attrs["desc"] = "These are the attribute labels of the image data having the same index."
            metadata.attrs["desc"] = "Metadata. Contains the COCO image ID of the image at the same index."

            # Now continue creating the HDF5 file with the list of attributes
            for i, ann in tqdm(enumerate(image_data), total=n_images):
                ann_id = ann["id"]
                ann_attrs = ann["attrs_vector"]

                # Get the object attributes as an array of 1s and 0s.
                img_attrs = np.array([np.float(x > 0) for x in ann_attrs])

                # Get the bounding box for the object
                bbox = ann['bbox']

                # Read the image data using the given path.
                try:
                    img = scp.ndimage.imread(ann["path"])
                except:
                    raise Exception("Invalid image path")

                # If single channel image, make it triple channeled
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))

                # Crop out the object whose attributes we have.
                # First we convert the coords to integers
                x, y, width, height = bbox

                # Now we crop out the object image
                # crop_img = img[y:y + height, x:x + width] - This is the original way to crop
                crop_img = common.get_image_crop(img, x, y, width, height, crop_size)

                # Resize it to the desired shape.
                img = scp.misc.imresize(crop_img, (crop_size, crop_size, 3))

                # Order the dimensions as expected (channels, width, height)
                img = np.transpose(img, (2, 0, 1))

                # We have all the required data so we now save it.
                dset[i, :, :, :] = img
                labels[i, :] = img_attrs
                metadata[i, :] = ann_id

        return True

    except (Exception,):
        traceback.print_exc()
        return False


def create_full_image_dataset(image_data, filename="train_fullscene.hdf5",
                              shape=(3, 224, 224), data_root=None):
    """
    Create a HDF5 file to store the training set, so we can use it with Keras or Blocks.
    This function stores the entire image in the HDF5 file without cropping out any segments.
    We store the images with the ordering in the first element of the shape tuple i.e. (idx, 3, x, y)
    Ref. https://www.getdatajoy.com/learn/Read_and_Write_HDF5_from_Python

    :param image_data: List of JSON objects with details about the images.
    :param filename: The filename of the HDF5 file.
    :param shape: The shape of the image to save into the HDF5 file.
    :param data_root: The root directory where to store the HDF5 file.
    :return: True if HDF5 file creation was successful, else False.
    """
    # Ensure the extension of the filename is `hdf5`
    if filename.split('.')[-1].strip() != 'hdf5':
        filename += ".hdf5"

    filename = os.path.join(data_root, filename)

    try:
        n_images = len(image_data)
        dataset_shape = (n_images,) + shape

        with h5py.File(filename, 'w') as f:

            dset = f.create_dataset("data", dataset_shape)
            labels = f.create_dataset("attributes", (n_images, 204))
            metadata = f.create_dataset("metadata", (n_images, 1))

            dset.attrs["desc"] = "This is the image data."
            labels.attrs["desc"] = "These are the attribute labels of the image data having the same index."
            metadata.attrs["desc"] = "Metadata. Contains the COCO image ID of the image at the same index."

            for i, ann in tqdm(enumerate(image_data), total=n_images):
                ann_id = ann["id"]
                ann_attrs = ann["attrs_vector"]

                # Get the object attributes as an array of 1s and 0s.
                img_attrs = np.array([np.float(x > 0) for x in ann_attrs])

                # Read the image data using the given path.
                try:
                    img = scp.ndimage.imread(ann["path"])
                except:
                    raise Exception("Invalid image path")

                # If single channel image, make it triple channeled
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))

                # Resize it to the desired shape.
                img = scp.misc.imresize(img, (shape[1], shape[1], 3))

                # Order the dimensions as expected (channels, width, height)
                img = np.transpose(img, (2, 0, 1))

                # We have all the required data so we now save it.
                dset[i, :, :, :] = img
                labels[i, :] = img_attrs
                metadata[i, :] = ann_id

        return True

    except (Exception,):
        traceback.print_exc()
        return False


def view_hdf5_image(hdf5_file, index):
    """
    Utility function to view the image at position `index` in the specified hdf5_file.
    Also prints out the attributes vector.
    :param hdf5_file: The HDF5 file from which to read the dataset.
    :param index: The position of the image in the HDF5 file you wish to view.
    :return: None
    """
    with h5py.File(hdf5_file) as f:
        dset = f["data"]
        attrs = f["attributes"]

        data = dset[index]
        attr_vector = attrs[index]
        image = np.transpose(data, (1, 2, 0))

        print(attr_vector)
        scp.misc.imshow(image)


if __name__ == "__main__":
    # Example of using the HDF5 helper functions.
    source_path = "../../../MSCOCO/annotations"

    from sklearn.externals import joblib
    import json
    import os.path as osp

    print("Reading the attributes")
    attr_data = joblib.load('../../../MSCOCO/cocottributes_new_version.jbl')

    print("Reading the annotation data")
    # train_ann = json.load(open(osp.join(source_path, 'instances_{0}2014.json'.format('train'))))
    val_ann = json.load(open(osp.join(source_path, 'instances_{0}2014.json'.format('val'))))

    print("Getting the formatted image list")
    root = "../../../MSCOCO"
    im_data_list = utils.get_images_list(val_ann['annotations'], attr_data, data_root=root, split="val")
    create_dataset(im_data_list, filename="test_sample.hdf5", data_root=root)
