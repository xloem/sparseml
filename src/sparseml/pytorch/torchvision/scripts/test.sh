#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

N_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l)+1))

#sparseml.image_classification.train

ROOT=$HOME/src/neuralmagic/sparseml/src/sparseml/pytorch/torchvision

python -m torch.distributed.run --nproc_per_node $N_GPUS $ROOT/train.py --recipe "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/pruned-conservative?recipe_type=original"   --dataset-path  /nm/drive1/ILSVRC2012/     --pretrained True     --arch-key resnet50     --batch-size 128     --workers 8     --output-dir sparsification_example/resnet50-imagenette-pruned     --save-best-after 8
