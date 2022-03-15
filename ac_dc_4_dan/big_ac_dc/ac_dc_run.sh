
python integrations/pytorch/train.py \
    --recipe-path recipe.yaml \
    --train-batch-size 256 \
    --test-batch-size 256 \
    --loader-num-workers 8 \
    --arch-key mobilenet \
    --save-epochs 195 295 395 495 595 \
    --pretrained False \
    --dataset imagenette \
    --optim-args '{"momentum": 0.875, "weight_decay": 0.00003051757813}' \
    --dataset-path /data
