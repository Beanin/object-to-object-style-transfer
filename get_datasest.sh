#!/usr/bin/env bash
DATASET_URL="http://images.cocodataset.org/zips/train2017.zip"
ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

DATASET_NAME="train2017"
ANNOTATIONS_DIR='annotations'

DATASET_ARCH="train2017.zip"
ANNOTATIONS_ARCH="annotations_trainval2017.zip"

if [[ ! -d $DATASET_NAME ]]; then
    wget $DATASET_URL || exit 2
    unzip $DATASET_ARCH || exit 3
    rm $DATASET_ARCH
fi

if [[ ! -d $ANNOTATIONS_DIR ]]; then
    wget $ANNOTATIONS_URL || exit 3
    unzip $ANNOTATIONS_ARCH || exit 4
    rm $ANNOTATIONS_ARCH
fi
