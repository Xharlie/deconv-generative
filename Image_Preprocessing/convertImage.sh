#!/usr/bin/env sh
TOOLS=/Users/xharlie/caffe/build/tools
RESIZE_HEIGHT=128
RESIZE_WIDTH=128
TRAIN_DATA_ROOT=/Users/xharlie/dev/deconv-generative/chair_processed/
SEGM_DATA_ROOT=/Users/xharlie/dev/deconv-generative/chair_segmented/
LABEL=/Users/xharlie/dev/deconv-generative/labels
DB_ROOT=/Users/xharlie/dev/deconv-generative/chair_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $TRAIN_DATA_ROOT \
    $LABEL/train.txt \
    $DB_ROOT/data_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $SEGM_DATA_ROOT \
    $LABEL/segm.txt \
    $DB_ROOT/segm_lmdb