#!/usr/bin/env sh
# Compute the mean image from the cocottributes training lmdb

EXAMPLE=/data/hays_lab/COCO/caffemodels_tmp/
TOOLS=/home/gen/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train-image-lmdb \
  cocottributes_mean.binaryproto

echo "Done."
