#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/pytorch/torchvision

MODEL=efficientnet_v2_s

#TRAIN_SIZE=300
TRAIN_SIZE=64

#EVAL_SIZE=384
EVAL_SIZE=64

EPOCHS=2

DATASET=/hdd/datasets/imagenette/imagenette/imagenette

sparseml.image_classification.train \
    --checkpoint-path zoo:cv/classification/efficientnet_v2-s/pytorch/sparseml/imagenet/base-none  \
    --arch-key $MODEL \
    --dataset-path $DATASET \
    --batch-size 128 \
    --lr 0.5 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-epochs 5 \
    --lr-warmup-method linear \
    --auto-augment imagenet \
    --epochs $EPOCHS \
    --random-erase 0.1 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.2 \
    --cutmix-alpha 1.0 \
    --weight-decay 0.00002 \
    --norm-weight-decay 0.0 \
    --train-crop-size $TRAIN_SIZE \
    --model-ema \
    --val-crop-size $EVAL_SIZE \
    --val-resize-size $EVAL_SIZE \
    --ra-sampler \
    --ra-reps 4

# python -m torch.distributed.launch --nproc_per_node=8 $ROOT/train.py \
# --model $MODEL --batch-size 128 --lr 0.5 --lr-scheduler cosineannealinglr \
# --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 600 --random-erase 0.1 \
# --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --weight-decay 0.00002 --norm-weight-decay 0.0 \
# --train-crop-size $TRAIN_SIZE --model-ema --val-crop-size $EVAL_SIZE --val-resize-size $EVAL_SIZE \
# --ra-sampler --ra-reps 4
