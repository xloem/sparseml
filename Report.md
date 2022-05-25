# End-to-end Manual Transformer Testing

## Question Answering

For testing, using the following stub.
```
stub = zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant_6layers-aggressive_96
```

### Sparsification Flow
Run the command below and train for few batches

```bash
sparseml.transformers.question_answering \
    --model_name_or_path /home/damian/.cache/sparsezoo/9768075d-2cf0-4bc7-98c3-fb6441a2ec36/pytorch \
    --dataset_name squad \
    --do_train \
    --output_dir './output' \
    --distill_teacher disable \ 
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni?recipe_type=transfer-question_answering \
```

**Observation**: lets add `onnxruntime` as a dependency of `transformers` or document in docs that user needs to do pip install `sparseml[torch, onnxruntime]`

### Test PyTorch Eval

```bash
sparseml.transformers.question_answering \
    --model_name_or_path /home/damian/.cache/sparsezoo/9768075d-2cf0-4bc7-98c3-fb6441a2ec36/pytorch \
    --dataset_name squad \
    --do_train \
    --output_dir './output' \
    --distill_teacher disable \ 
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni?recipe_type=transfer-question_answering \
```
**FAILS**

### Test Onnx Export

```bash
sparseml.transformers.export_onnx   --task question-answering --model_path /home/damian/.cache/sparsezoo/9768075d-2cf0-4bc7-98c3-fb6441a2ec36/pytorch
```

### Test DeepSparse Eval

```bash
deepsparse.transformers.eval_downstream \
     /home/damian/.cache/sparsezoo/9768075d-2cf0-4bc7-98c3-fb6441a2ec36/pytorch \
    --dataset squad
```

### Test Pipeline
Completed

### Test Serv
Completed





## Text Classification

For testing, using the following stub.
```
stub = zoo:nlp/text_classification/bert-base/pytorch/huggingface/mnli/12layer_pruned90-none
```

### Sparsification Flow

Run the command below and train for few batches

```bash
sparseml.transformers.text_classification \
    --model_name_or_path /home/damian/.cache/sparsezoo/4422c5be-f1bf-4659-a28f-06acb18c6308/pytorch \
    --task_name mnli \
    --do_train \
    --do_eval \
    --output_dir models/teacher \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none?recipe_type=transfer-text_classification
```

### Test PyTorch Eval

```bash
sparseml.transformers.text_classification \
    --model_name_or_path /home/damian/.cache/sparsezoo/4422c5be-f1bf-4659-a28f-06acb18c6308/pytorch \
    --task_name mnli \
    --do_eval \
    --output_dir models/teacher \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none?recipe_type=transfer-text_classification
```


### Test Onnx Export

```bash
sparseml.transformers.export_onnx   --task text-classification --model_path /home/damian/.cache/sparsezoo/4422c5be-f1bf-4659-a28f-06acb18c6308/pytorch
```

### Test DeepSparse Eval

```bash
deepsparse.transformers.eval_downstream \
     /home/damian/.cache/sparsezoo/4422c5be-f1bf-4659-a28f-06acb18c6308/pytorch \
    --dataset mnli
```

### Test Pipeline
Completed

### Test Serv
Completed

## Token Classification

For testing, using the following stub.
```
stub = zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni
```

### Sparsification Flow

Run the command below and train for few batches

```bash
sparseml.transformers.token_classification 
--model_name_or_path /home/damian/.cache/sparsezoo/1f657548-ae05-4706-8ca4-b3810d04db2c/pytorch --dataset_name conll2003 --do_train --do_eval --output_dir './output' --distill_teacher disable --recipe zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni --overwrite_output_dir --per_device_train_batch_size 2 --per_device_eval_batch_size 2
```

FAILS

### Test DeepSparse Eval

```bash
deepsparse.transformers.eval_downstream \
     /home/damian/.cache/sparsezoo/1f657548-ae05-4706-8ca4-b3810d04db2c/pytorch \
    --dataset conll2003
```



## Test Onnx Export

```bash
sparseml.transformers.export_onnx   --task token-classification --model_path /home/damian/.cache/sparsezoo/1f657548-ae05-4706-8ca4-b3810d04db2c/pytorch
```

### Test Pipeline
According to docs

### Test Serv
According to docs

### Test DeepSparse Eval
No dataset to test

## MLM

For testing, using the following stub.
```
stub = zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/6layer_pruned80_quant-none-vnni
```

### Sparsification Flow

Run the command below and train for few batches

```bash
sparseml.transformers.train.text_classification \
  --output_dir sparse_quantized_bert-text_classification_mnli \
  --model_name_or_path /home/damian/.cache/sparsezoo/948d6023-d23c-4dd9-992f-5a560bc5e206/pytorch \
  --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/6layer_pruned80_quant-none-vnni?recipe_type=transfer-text_classification \
  --distill_teacher zoo:nlp/text_classification/bert-base/pytorch/huggingface/mnli/base-none \
  --task_name mnli \
  --do_train \
  --do_eval  
```

# Testing docs
- https://github.com/neuralmagic/sparseml/tree/main/integrations/huggingface-transformers
1 the output is different, example fails
- https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers








