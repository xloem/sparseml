#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

ROOT=/hdd/src/neuralmagic/sparseml/src/sparseml/pytorch/torchvision

MODEL=efficientnet_v2_s

EVAL_SIZE=384

DATASET=/hdd/datasets/ILSVRC

sparseml.image_classification.train \
    --checkpoint-path zoo:cv/classification/efficientnet_v2-s/pytorch/sparseml/imagenet/base-none \
    --arch-key efficientnet_v2_s \
    --dataset-path $DATASET \
    --test-only \
    --val-resize-size $EVAL_SIZE \
    --val-crop-size $EVAL_SIZE \
    --batch-size 16 \
    --workers 6
