#!/bin/bash
CAFFE_HOME=/home/ubuntu/caffe-nsight/caffe

echo "Create train lmdb.."
rm -rf img_train_lmdb/
$CAFFE_HOME/build/tools/convert_multilabel \
--shuffle \
--resize_height=480 \
--resize_width=640 \
train/ \
train.txt \
img_train_lmdb \
img_train_label_lmdb \
2

echo "Create test lmdb.."
rm -rf img_test_lmdb
$CAFFE_HOME/build/tools/convert_multilabel \
--shuffle \
--resize_height=480 \
--resize_width=640 \
test/ \
test.txt \
img_test_lmdb \
img_test_label_lmdb \
2

echo "All Done.."
