# Contrastive Loss with Text Classification

## Generate Sample Dataset
Generate sample dataset by running `python download_sst2.py`. Generates full validation CSV
and sample of 8 training examples per class by default.

## Example Command

```bash
sparseml.transformers.train.text_classification \
  --output_dir contrastive-loss-example \
  --model_name_or_path zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned80-none-vnni \
  --recipe zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned80_quant-none-vnni \
  --max_seq_length 128 --per_device_train_batch_size 4 --per_device_eval_batch_size 32 --preprocessing_num_workers 6 \
  --do_train --do_eval --evaluation_strategy epoch --fp16  \
  --save_strategy epoch --save_total_limit 1 \
  --distill_teacher disable \
  --train_file sst2-train-8-samples.csv \
  --validation_file sst2-validation.csv \
  --input_column_names sentence \
  --label_column_name label
```
