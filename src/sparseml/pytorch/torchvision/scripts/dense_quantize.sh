#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2

N_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l)+1))

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/pytorch/torchvision

DST_MODEL_DIR=/nm/drive3/tuan/models/efficientnet/ongoing

RECIPE_DIR=$ROOT/recipes
RECIPE_NAME=dense_quantize_v2

SRC_MODEL_NAME=efficientnet_v2_s

LOGGING_STEPS=200
EVAL_STEPS=400

TRAIN_SIZE=300
EVAL_SIZE=384

DATASET=/nm/drive1/ILSVRC
#DATASET=/nm/drive1/ILSVRC2012/imagenette-160
BSIZE_PER_GPU=32
GRAD_ACCUM=4

for NUM_EPOCHS in 3
do
for LR in 1e-5
do

ID=$RANDOM
OPT=rmsprop
DST_MODEL_NAME=$SRC_MODEL_NAME@$RECIPE_NAME@EP$NUM_EPOCHS@OPT$OPT@LR$LR@ID$ID

RECIPE_ARGS="{\"num_epochs\":${NUM_EPOCHS},\"warmup_lr\":${LR}}"

export WANDB_NAME=$DST_MODEL_NAME

python -m torch.distributed.run --nproc_per_node $N_GPUS $ROOT/train.py \
    --checkpoint-path zoo:cv/classification/efficientnet_v2-s/pytorch/sparseml/imagenet/base-none  \
    --arch-key $SRC_MODEL_NAME \
    --output-dir $DST_MODEL_DIR/$DST_MODEL_NAME \
    --recipe $RECIPE_DIR/$RECIPE_NAME.md \
    --recipe_args $RECIPE_ARGS \
    --dataset-path $DATASET \
    --opt $OPT \
    --batch-size $BSIZE_PER_GPU \
    --gradient-accum-steps $GRAD_ACCUM \
    --auto-augment imagenet \
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
    --ra-reps 4 \
    --workers 64 \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS

done
done

