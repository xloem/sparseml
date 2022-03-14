
python integrations/pytorch/train.py \
    --recipe-path ac_dc_recipe.md \
    --train-batch-size 256 \
    --test-batch-size 256 \
    --loader-num-workers 8 \
    --arch-key mobilenet \
    --pretrained False \
    --dataset imagenette \
    --optim-args '{"momentum": 0.875, "weight_decay": 0.00003051757813}' \
    --dataset-path /data
